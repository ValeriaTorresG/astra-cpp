"""
build the data + random catalog for one iteration.

all fits i/o, random pool sampling, and cartesian coordinate conversion happen
in c++ (`_core`); this module only assembles the arrays.
"""
from __future__ import annotations

import glob
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _core
from .config import GHOST_SPACINGS, RANDOM_ID_FACTOR, CubicConfig, CutskyConfig
from .monitor import ResourceMonitor, gib


# position columns for abacussummit boxes (mpc/h)
CUBIC_COLUMNS = ('x', 'y', 'z')


@dataclass
class PeriodicBox:
    lo: list[float]
    hi: list[float]
    ghost_margin: float


@dataclass
class Catalog:
    """points for one iteration. the first `n_data` rows are always the real sample."""
    targetid: np.ndarray                 # int64; real objects > 0, randoms < 0
    is_data: np.ndarray                  # bool
    xyz: np.ndarray | None               # (n, 3) float64, released after triangulation
    n_data: int
    columns: dict[str, np.ndarray] = field(default_factory=dict)  # extra columns to save
    meta: dict = field(default_factory=dict)
    box: PeriodicBox | None = None       # periodic box (full cubic boxes only)

    @property
    def size(self) -> int:
        return len(self.targetid)


@dataclass
class RandomSource:
    seed: int
    table: _core.FitsTable

    @property
    def path(self) -> str:
        return self.table.path

    @property
    def nrows(self) -> int:
        return self.table.nrows


def seed_of(path: Path) -> int:
    match = re.fullmatch(r'random_S(\d+)_\d+X\.fits', path.name)
    if not match:
        raise ValueError(f'cannot extract the seed from {path.name}')
    return int(match.group(1))


def discover_randoms(pattern: str, required: list[str]) -> list[RandomSource]:
    paths = sorted((Path(p) for p in glob.glob(pattern)), key=seed_of)
    if not paths:
        raise FileNotFoundError(f'no files match {pattern}')
    sources = []
    for path in paths:
        table = _core.FitsTable(str(path))
        missing = [c for c in required if not table.has_column(c)]
        if missing:
            raise ValueError(f'{path.name} is missing columns {missing}')
        if table.nrows >= RANDOM_ID_FACTOR:
            raise ValueError(f'{path.name} has too many rows for stable TARGETIDs')
        sources.append(RandomSource(seed_of(path), table))
    return sources


def load_cutsky(cfg: CutskyConfig, log: logging.Logger, mon: ResourceMonitor) -> Catalog:
    """real sample + an equally sized random pool, sampled uniformly without
    replacement from the union of all seed files. with `cfg.footprint`, both the
    real sample and the pool only contain rows where that column is nonzero."""
    columns = ['RA', 'DEC', cfg.redshift]
    required = columns + ([cfg.footprint] if cfg.footprint else [])
    real = _core.FitsTable(str(cfg.real_file))
    missing = [c for c in required if not real.has_column(c)]
    if missing:
        raise ValueError(f'{cfg.real_file.name} is missing columns {missing}')
    sources = discover_randoms(cfg.random_glob, required)

    # rows inside the footprint (None: all rows); they keep their file row numbers
    if cfg.footprint:
        with mon.stage('footprint'):
            selected = _core.select_rows([real] + [s.table for s in sources], cfg.footprint, threads=cfg.threads)
    else:
        selected = [None] * (1 + len(sources))
    real_sel, random_sel = selected[0], selected[1:]
    del selected
    n_avail = real.nrows if real_sel is None else len(real_sel)
    n_real = min(n_avail, cfg.max_real_rows) if cfg.max_real_rows else n_avail
    if n_real == 0:
        raise ValueError(f'{cfg.real_file.name} has no rows in the footprint {cfg.footprint}')
    if real_sel is None:
        real_rows = None if n_real == real.nrows else np.arange(n_real, dtype=np.int64)
    else:
        real_rows = real_sel[:n_real]
    sizes = [s.nrows if sel is None else len(sel) for s, sel in zip(sources, random_sel)]
    pool_size = sum(sizes)
    log.info('footprint: %s', cfg.footprint or 'none (full catalog)')
    log.info('real: %s rows, %s in footprint (using %s) | %s', real.nrows, n_avail, n_real, cfg.real_file)
    log.info('random pool: %s files, %s rows in the footprint of %s (%.1fx the real sample)',
             len(sources), pool_size, sum(s.nrows for s in sources), pool_size / n_real)

    with mon.stage('pool_randoms'):
        picked = _core.sample_pool(sizes, n_real, cfg.seed)
    rows = [p if sel is None else sel[p] for p, sel in zip(picked, random_sel)]
    del picked, random_sel
    for source, r, size in zip(sources, rows, sizes):
        log.info('  random seed=%s sorteadas=%s/%s', source.seed, len(r), size)

    n = 2 * n_real
    ra, dec, z = (np.empty(n, dtype=np.float64) for _ in range(3))
    requests = [(real, real_rows, 0)]
    offset = n_real
    for source, r in zip(sources, rows):
        requests.append((source.table, r, offset))
        offset += len(r)
    with mon.stage('lectura_fits'):
        _core.read_columns(requests, columns, [ra, dec, z], threads=cfg.threads)

    # real targetid = one-based row in the real file
    targetid = np.empty(n, dtype=np.int64)
    targetid[:n_real] = np.arange(1, n_real + 1, dtype=np.int64) if real_rows is None else real_rows + 1
    offset = n_real
    for source, r in zip(sources, rows):
        targetid[offset:offset + len(r)] = -(source.seed * RANDOM_ID_FACTOR + r + 1)
        offset += len(r)
    is_data = np.zeros(n, dtype=bool)
    is_data[:n_real] = True

    with mon.stage('coordenadas'):
        xyz = _core.sky_to_cartesian(ra, dec, z, _core.FlatLambdaCDM.planck18(), threads=cfg.threads)

    meta = {'catalog': 'cutsky', 'footprint': cfg.footprint, 'real_file_rows': real.nrows,
            'n_real': n_real, 'n_random': n_real, 'pool_rows': pool_size,
            'random_sources': [{'seed': s.seed, 'path': s.path, 'rows': s.nrows, 'in_footprint': size,
                                'sampled': len(r)}
                               for s, r, size in zip(sources, rows, sizes)],
            'cosmology': 'Planck18 (FlatLambdaCDM, astropy)', 'distance_unit': 'Mpc'}
    return Catalog(targetid, is_data, xyz, n_real, {'RA': ra, 'DEC': dec, 'Z': z}, meta)


def _box_bounds(xyz_data: np.ndarray, box_size: float) -> tuple[np.ndarray, np.ndarray]:
    """abacussummit [lo, lo + l)^3 box: centered at 0 or with a corner at 0."""
    lo_data, hi_data = float(xyz_data.min()), float(xyz_data.max())
    origin = 0.0 if lo_data >= 0 else -box_size / 2
    lo, hi = np.full(3, origin), np.full(3, origin + box_size)
    tol = 1e-6 * box_size
    if lo_data < origin - tol or hi_data > origin + box_size + tol:
        raise ValueError(f'galaxies in [{lo_data}, {hi_data}] are outside the box '
                         f'[{origin}, {origin + box_size}]; check --box-size')
    return lo, hi


def _target_ids(n_data: int, n_random: int, data_ids: np.ndarray | None = None) -> np.ndarray:
    """real objects: file row + 1. randoms: -(1..n_random). no large temporaries."""
    targetid = np.arange(1, n_data + n_random + 1, dtype=np.int64)
    if data_ids is not None:
        targetid[:n_data] = data_ids
    rand = targetid[n_data:]
    rand -= n_data
    np.negative(rand, out=rand)
    return targetid


def load_cubic(cfg: CubicConfig, log: logging.Logger, mon: ResourceMonitor) -> Catalog:
    """abacussummit cubic box + uniform randoms within the box.

    full box (periodic): read x, y, z directly into the final position array,
    wrap them into [lo, hi), then generate randoms in place without intermediate
    copies. --subbox selects a nonperiodic subbox for tests.
    """
    table = _core.FitsTable(str(cfg.box_file))
    missing = [c for c in CUBIC_COLUMNS if not table.has_column(c)]
    if missing:
        raise ValueError(f'{cfg.box_file.name} is missing columns {missing}')
    n_all = table.nrows

    if cfg.subbox:
        everything = np.empty((n_all, 3), dtype=np.float64)
        with mon.stage('lectura_fits'):
            _core.read_columns([(table, None, 0)], list(CUBIC_COLUMNS),
                               [everything[:, k] for k in range(3)], threads=cfg.threads)
        lo, hi = _box_bounds(everything, cfg.box_size)
        hi = lo + cfg.subbox
        index = np.flatnonzero(np.all((everything >= lo) & (everything < hi), axis=1))
        n_data = len(index)
        n_random = int(round(cfg.random_factor * n_data))
        xyz = np.empty((n_data + n_random, 3), dtype=np.float64)
        xyz[:n_data] = everything[index]
        del everything
        data_ids, periodic = index + 1, False
    else:
        n_data = n_all
        n_random = int(round(cfg.random_factor * n_data))
        xyz = np.empty((n_data + n_random, 3), dtype=np.float64)
        log.info('posiciones: (%s, 3) float64 = %.1f GiB', len(xyz), xyz.nbytes / 2**30)
        with mon.stage('lectura_fits'):
            _core.read_columns([(table, None, 0)], list(CUBIC_COLUMNS),
                               [xyz[:, k] for k in range(3)], threads=cfg.threads)
        lo, hi = _box_bounds(xyz[:n_data], cfg.box_size)
        data_ids, periodic = None, cfg.periodic
        if periodic:
            _core.wrap_periodic(xyz[:n_data], lo.tolist(), hi.tolist(), threads=cfg.threads)

    log.info('caja: %s galaxias (de %s) en [%s, %s)^3 | randoms uniformes: %s | periódica: %s',
             n_data, n_all, lo[0], hi[0], n_random, periodic)
    with mon.stage('randoms_caja'):
        _core.fill_uniform_box(xyz[n_data:], lo.tolist(), hi.tolist(), cfg.seed, threads=cfg.threads)

    targetid = _target_ids(n_data, n_random, data_ids)
    is_data = np.zeros(n_data + n_random, dtype=bool)
    is_data[:n_data] = True

    box = None
    meta = {'catalog': 'cubic', 'n_real': n_data, 'n_random': n_random, 'box_rows': n_all,
            'box_lo': lo.tolist(), 'box_hi': hi.tolist(), 'periodic': periodic, 'distance_unit': 'Mpc/h'}
    if periodic:
        spacing = (np.prod(hi - lo) / len(xyz)) ** (1 / 3)
        margin = cfg.ghost_margin if cfg.ghost_margin > 0 else GHOST_SPACINGS * spacing
        box = PeriodicBox(lo.tolist(), hi.tolist(), float(margin))
        meta.update(ghost_margin=float(margin), mean_spacing=float(spacing))
        log.info('periodicidad: margen de fantasmas %.2f (%.1f separaciones medias de %.3f)',
                 margin, margin / spacing, spacing)
    return Catalog(targetid, is_data, xyz, n_data, {}, meta, box)

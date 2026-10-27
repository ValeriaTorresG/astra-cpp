"""Una iteración de ASTRA: catálogo -> Delaunay (CGAL) -> NDATA/NRAND -> r -> WEBTYPE."""
from __future__ import annotations

import datetime as dt
import logging
import sys
import time
from typing import Callable

import numpy as np

from . import _core
from .catalogs import Catalog, load_cubic, load_cutsky
from .config import WEBTYPES, AstraParams, CubicConfig, CutskyConfig, to_jsonable
from .monitor import ResourceMonitor, gib, host_info, mem_available, peak_rss_bytes, setup_logger
from .outputs import (CLASS_DTYPE, RunPaths, aggregate, iteration_complete, points_dtype,
                      write_json, write_structured)

Loader = Callable[[AstraParams, logging.Logger, ResourceMonitor], Catalog]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


# Pico medido en la corrida cut-sky completa (93.6 GiB para 127.4M puntos), incluyendo
# los arreglos de Python; sirve para avisar antes de triangular si no va a caber.
PEAK_BYTES_PER_POINT = 800


def check_memory(cat: Catalog, periodic: dict, log: logging.Logger) -> None:
    n = cat.size
    if periodic:
        side = np.subtract(periodic['box_hi'], periodic['box_lo'])
        n *= float(np.prod(1 + 2 * periodic['ghost_margin'] / side))
    need = PEAK_BYTES_PER_POINT * n
    available = mem_available()
    log.info('memoria estimada para la triangulación: %s (disponible en el nodo: %s)',
             gib(need), gib(available) if available else '?')
    if available and need > 0.95 * available:
        log.warning('la triangulación probablemente no quepa en memoria: usa un nodo con más RAM, '
                    '--random-factor menor o --subbox')


def classify_catalog(cat: Catalog, params: AstraParams, log: logging.Logger,
                     mon: ResourceMonitor) -> tuple[dict[str, np.ndarray], dict]:
    """Triangula todos los puntos (data + randoms) y clasifica cada uno.

    Libera `cat.xyz` y la triangulación en cuanto dejan de hacer falta.
    """
    periodic = {}
    if cat.box is not None:
        periodic = {'box_lo': cat.box.lo, 'box_hi': cat.box.hi, 'ghost_margin': cat.box.ghost_margin}
        log.info('caja periódica [%s, %s)^3: copias fantasma a menos de %.2f de cada cara',
                 cat.box.lo[0], cat.box.hi[0], cat.box.ghost_margin)
    check_memory(cat, periodic, log)
    log.info('Delaunay global: %s puntos (%s data + %s random) threads=%s',
             cat.size, cat.n_data, cat.size - cat.n_data, params.tri_threads)
    with mon.stage('triangulacion'):
        tri = _core.Delaunay3(cat.xyz, threads=params.tri_threads, lock_grid_cells=params.lock_grid_cells,
                              **periodic)
    cat.xyz = None
    stats = {'vertices': int(tri.number_of_vertices()),
             'finite_cells': int(tri.number_of_finite_cells()),
             'parallel_triangulation': bool(tri.parallel),
             'lock_grid_cells': int(tri.lock_grid_cells),
             'duplicates': int(len(tri.duplicates)),
             'ghosts': int(tri.n_ghosts), **periodic}
    log.info('triangulación: vertices=%s (fantasmas=%s) tetraedros=%s paralela=%s lock_grid=%s^3',
             stats['vertices'], stats['ghosts'], stats['finite_cells'], stats['parallel_triangulation'],
             stats['lock_grid_cells'])
    if stats['duplicates']:
        log.warning('%s puntos duplicados exactos: heredan los conteos de su vértice', stats['duplicates'])

    with mon.stage('vecinos'):
        ndata, nrand = tri.neighbor_counts(cat.is_data, threads=params.threads)
    if periodic:
        stats['unverified'] = int(tri.n_unverified)
        if tri.n_unverified:
            raise RuntimeError(f'{tri.n_unverified} puntos tienen vecinos que dependen de puntos fuera de la '
                               f'capa periódica de {periodic["ghost_margin"]:.2f}: aumenta --ghost-margin')
        log.info('vecinos periódicos verificados para los %s puntos', cat.size)
    del tri

    with mon.stage('clasificacion'):
        r, webtype = _core.classify(ndata, nrand, params.r_lower, params.r_med, params.r_upper,
                                    threads=params.threads)
    stats['mean_degree'] = float(ndata.mean(dtype=np.float64) + nrand.mean(dtype=np.float64))
    stats['max_degree'] = int(np.max(ndata + nrand))
    for label, sl in (('data', slice(0, cat.n_data)), ('random', slice(cat.n_data, None))):
        counts = np.bincount(webtype[sl], minlength=4)
        stats[f'webtype_{label}'] = dict(zip(WEBTYPES, counts.tolist()))
        log.info('clases %s: %s', label,
                 ' '.join(f'{w}={c / max(1, counts.sum()):.4f}' for w, c in zip(WEBTYPES, counts)))
    columns = {'TARGETID': cat.targetid, 'ISDATA': cat.is_data, 'NDATA': ndata, 'NRAND': nrand,
               'R': r, 'WEBTYPE': webtype}
    return columns, stats


def save_points(cat: Catalog, path) -> None:
    extra = list(cat.columns)
    columns = {'TARGETID': cat.targetid, 'ISDATA': cat.is_data, **cat.columns,
               'XCART': cat.xyz[:, 0], 'YCART': cat.xyz[:, 1], 'ZCART': cat.xyz[:, 2]}
    write_structured(path, columns, points_dtype(extra))


def run_iteration(cfg: AstraParams, loader: Loader) -> dict:
    cfg.validate()
    paths = RunPaths(cfg.run_dir)
    paths.make_dirs()
    it = cfg.iteration
    log = setup_logger(paths.log(it))
    log.info('astra-cpp | iteración %s | %s', it, type(cfg).__name__)
    log.info('host=%s | python %s | numpy %s | CGAL %s | TBB=%s | threads=%s',
             host_info()['host'], sys.version.split()[0], np.__version__, _core.CGAL_VERSION,
             _core.HAS_PARALLEL_TRIANGULATION, cfg.threads)
    log.info('salida: %s', paths.root)

    if not cfg.overwrite and iteration_complete(paths, it):
        log.info('iteración %s ya completa; se omite (usa --overwrite para repetirla)', it)
        if cfg.aggregate:
            aggregate(paths, cfg.threads, log)
        return {'status': 'skipped'}

    summary = {'status': 'running', 'iteration': it, 'seed': cfg.seed, 'started_at': _now(),
               'config': to_jsonable(cfg), 'host': host_info(), 'cgal': _core.CGAL_VERSION}
    write_json(paths.summary(it), summary)
    t0 = time.monotonic()
    with ResourceMonitor(log, cfg.log_interval) as mon:
        try:
            cat = loader(cfg, log, mon)
            summary.update(cat.meta)
            summary['n_points'] = cat.size
            if cfg.save_points:
                with mon.stage('guardar_puntos'):
                    save_points(cat, paths.points(it))
            cat.columns.clear()
            columns, stats = classify_catalog(cat, cfg, log, mon)
            with mon.stage('guardar_clasificacion'):
                write_structured(paths.classification(it), columns, CLASS_DTYPE)
            del columns
            summary.update(status='complete', completed_at=_now(), stats=stats,
                           elapsed_seconds=round(time.monotonic() - t0, 2),
                           timings_seconds={k: round(v, 3) for k, v in mon.timings.items()},
                           peak_rss_gib=round(peak_rss_bytes() / 2**30, 2),
                           classification=str(paths.classification(it)),
                           points=str(paths.points(it)) if cfg.save_points else None)
            write_json(paths.summary(it), summary)
            log.info('iteración %s completa en %.1f s | pico de memoria %.2f GiB',
                     it, summary['elapsed_seconds'], summary['peak_rss_gib'])
        except BaseException as exc:
            log.exception('iteración %s falló', it)
            summary.update(status='failed', failed_at=_now(), error=repr(exc))
            write_json(paths.summary(it), summary)
            raise
        if cfg.aggregate:
            with mon.stage('agregacion'):
                summary['aggregate'] = aggregate(paths, cfg.threads, log)
    return summary


def run_cutsky(cfg: CutskyConfig) -> dict:
    return run_iteration(cfg, load_cutsky)


def run_cubic(cfg: CubicConfig) -> dict:
    return run_iteration(cfg, load_cubic)

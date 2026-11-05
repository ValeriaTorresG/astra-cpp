"""consensus void catalog over several realizations (astra-desi consensus.py).

pool the clean voids (BORDER == False, finite shape) of every completed
iteration, group them largest-first by contained volume fraction, take the
lower-median-radius vote of each group as its representative, prune overlapping
representatives, and keep those supported by more than v_cut of the iterations.
the grouping runs in c++ (`_core.void_consensus`).
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .. import _core
from ..outputs import read_json, write_json, write_structured
from .config import DEFAULT_CONSENSUS_V_CUT, DEFAULT_CONSENSUS_VOL_FRAC
from .outputs import CATALOG_DTYPE, CONSENSUS_DTYPE, SHAPE_FIELDS, VoidPaths, completed_iterations, iteration_complete


def pool_voids(paths: VoidPaths, iterations: list[int]) -> dict:
    """aligned arrays of the usable voids of `iterations` (origin = position in the list)."""
    parts, counts = [], {'n_read': 0, 'n_border': 0, 'n_undefined_shape': 0}
    for index, it in enumerate(iterations):
        cat = np.load(paths.catalog(it))
        if cat.dtype != CATALOG_DTYPE:
            raise ValueError(f'{paths.catalog(it)} does not have the void catalog dtype')
        finite = np.isfinite(cat['R_EFF']) & np.isfinite(cat['ELLIP'])
        for name in SHAPE_FIELDS + ('XCART', 'YCART', 'ZCART'):
            finite &= np.isfinite(cat[name])
        usable = finite & ~cat['BORDER']
        counts['n_read'] += len(cat)
        counts['n_border'] += int(np.count_nonzero(cat['BORDER']))
        counts['n_undefined_shape'] += int(np.count_nonzero(~finite))
        parts.append((cat[usable], np.full(int(np.count_nonzero(usable)), index, dtype=np.int32)))
    pooled = np.concatenate([p for p, _ in parts]) if parts else np.zeros(0, dtype=CATALOG_DTYPE)
    origin = np.concatenate([o for _, o in parts]) if parts else np.zeros(0, dtype=np.int32)
    if np.any(pooled['R_EFF'] <= 0):
        raise ValueError('usable voids must have R_EFF > 0')
    return {'voids': pooled, 'origin': origin, **counts}


def periodic_box(paths: VoidPaths, iterations: list[int]) -> tuple[list[float], list[float]] | None:
    """box of periodic runs in catalog units (mpc/h), None otherwise; all iterations must agree."""
    boxes = set()
    for it in iterations:
        summary = read_json(paths.summary(it)) or {}
        if summary.get('periodic'):
            scale = float(summary.get('stats', {}).get('coordinate_scale', 1.0))
            boxes.add((tuple(np.multiply(summary['box_lo'], scale)), tuple(np.multiply(summary['box_hi'], scale))))
        else:
            boxes.add(None)
    if len(boxes) != 1:
        raise ValueError('the iterations mix periodic and nonperiodic runs or different boxes')
    box = boxes.pop()
    return None if box is None else (list(box[0]), list(box[1]))


def build_consensus(pool: dict, iterations: list[int], vol_frac: float = DEFAULT_CONSENSUS_VOL_FRAC,
                    v_cut: float = DEFAULT_CONSENSUS_V_CUT, keep_all: bool = False,
                    box: tuple[list[float], list[float]] | None = None, threads: int = 0) -> tuple[np.ndarray, dict]:
    voids = pool['voids']
    if len(voids) == 0:
        raise ValueError('no non-border voids with finite shapes were found')
    centers = np.column_stack([voids['XCART'], voids['YCART'], voids['ZCART']])
    box_args = {'box_lo': box[0], 'box_hi': box[1]} if box else {}
    res = _core.void_consensus(centers, np.ascontiguousarray(voids['R_EFF']), pool['origin'], len(iterations),
                               vol_frac=vol_frac, threads=threads, **box_args)
    rows = res['rows']
    out = np.zeros(len(rows), dtype=CONSENSUS_DTYPE)
    picked = voids[rows]
    out['VOID_ID'] = picked['VOID_ID']
    out['SRC_ITER'] = np.asarray(iterations, dtype=np.int32)[pool['origin'][rows]]
    out['X'], out['Y'], out['Z'] = picked['XCART'], picked['YCART'], picked['ZCART']
    for name in ('R_EFF', 'ELLIP') + SHAPE_FIELDS:
        out[name] = picked[name]
    out['V'] = res['support']
    out['FRAC_V'] = res['support'] / float(len(iterations))
    out['N_ABSORBED'] = res['n_absorbed']
    n_after_pruning = len(out)
    if not keep_all:
        out = out[out['FRAC_V'] > v_cut]
    if len(np.unique(out['VOID_ID'])) != len(out):
        raise RuntimeError('the consensus produced duplicate VOID_IDs')
    stats = {'n_pooled': int(len(voids)), 'n_groups': int(res['n_groups']), 'n_after_pruning': int(n_after_pruning),
             'n_after_support_cut': int(len(out)), 'n_pruning_duplicates': int(res['n_duplicates'])}
    return out, stats


def run_consensus(paths: VoidPaths, iterations: list[int] | None = None, vol_frac: float = DEFAULT_CONSENSUS_VOL_FRAC,
                  v_cut: float = DEFAULT_CONSENSUS_V_CUT, keep_all: bool = False, threads: int = 0,
                  log: logging.Logger | None = None) -> dict | None:
    """consensus over `iterations` (default: every completed iteration); writes
    consensus/voids_nNNN[_all].npy and .json."""
    log = log or logging.getLogger('astra_cpp.voids')
    if iterations is None:
        iterations = completed_iterations(paths)
    else:
        iterations = list(dict.fromkeys(int(i) for i in iterations))
        missing = [it for it in iterations if not iteration_complete(paths, it)]
        if missing:
            raise FileNotFoundError(f'incomplete void iterations in {paths.root}: {missing}')
    if not iterations:
        log.info('consensus skipped: no completed void iterations')
        return None
    params = {it: (read_json(paths.summary(it)) or {}).get('finder_params') for it in iterations}
    if len({repr(v) for v in params.values()}) != 1:
        raise ValueError('the iterations were produced with different void finder parameters')

    t0 = time.monotonic()
    log.info('consensus over %s iterations: %s', len(iterations), iterations)
    pool = pool_voids(paths, iterations)
    box = periodic_box(paths, iterations)
    catalog, stats = build_consensus(pool, iterations, vol_frac, v_cut, keep_all, box, threads)
    npy = paths.consensus(len(iterations), keep_all)
    write_structured(npy, {n: catalog[n] for n in CONSENSUS_DTYPE.names}, CONSENSUS_DTYPE)
    summary = {'algorithm': 'six-step astra consensus catalogue', 'iterations': iterations,
               'n_iterations': len(iterations), 'vol_frac': vol_frac, 'v_cut': None if keep_all else v_cut,
               'support_policy': 'all post-pruning representatives' if keep_all else 'strictly FRAC_V > v_cut',
               'representative': 'lower median-radius vote; one largest vote per run',
               'periodic_box': box, 'finder_params': next(iter(params.values())),
               'n_voids_read': pool['n_read'], 'n_border': pool['n_border'],
               'n_undefined_shape': pool['n_undefined_shape'], **stats, 'path': str(npy),
               'elapsed_seconds': round(time.monotonic() - t0, 3)}
    if len(catalog):
        q = np.percentile(catalog['R_EFF'], (16, 50, 84))
        summary['catalog_statistics'] = {'median_r_eff': float(q[1]), 'r_eff_p16': float(q[0]),
                                         'r_eff_p84': float(q[2]), 'median_ellip': float(np.median(catalog['ELLIP']))}
    write_json(paths.consensus(len(iterations), keep_all, 'json'), summary)
    log.info('consensus: %s pooled -> %s representatives -> %s after pruning -> %s saved to %s',
             stats['n_pooled'], stats['n_groups'], stats['n_after_pruning'], stats['n_after_support_cut'], npy)
    return summary

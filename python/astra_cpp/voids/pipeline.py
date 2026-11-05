"""one void-finder iteration: same catalog as the classification iteration
(same loaders and seed), then delaunay -> watershed -> selection -> shapes."""
from __future__ import annotations

import sys
import time

import numpy as np

from .. import _core
from ..catalogs import load_cubic, load_cutsky
from ..config import to_jsonable
from ..monitor import ResourceMonitor, host_info, peak_rss_bytes, setup_logger
from ..outputs import RunPaths, points_dtype, valid_npy, write_json, write_structured
from ..pipeline import Loader, _now, save_points
from .config import VoidCubicConfig, VoidCutskyConfig
from .consensus import run_consensus
from .finder import find_voids
from .outputs import CATALOG_DTYPE, CLEAN_DTYPE, MEMBERSHIP_DTYPE, VoidPaths, iteration_complete


def run_void_iteration(cfg, loader: Loader) -> dict:
    cfg.validate()
    paths = VoidPaths(cfg.void_dir)
    paths.make_dirs()
    it = cfg.iteration
    log = setup_logger(paths.log(it), name='astra_cpp.voids')
    log.info('astra-cpp voids | iteration %s | %s', it, type(cfg).__name__)
    log.info('host=%s | python %s | numpy %s | cgal %s | tbb=%s | threads=%s',
             host_info()['host'], sys.version.split()[0], np.__version__, _core.CGAL_VERSION,
             _core.HAS_PARALLEL_TRIANGULATION, cfg.threads)
    log.info('output: %s', paths.root)
    finder_params = cfg.finder_params()

    if not cfg.overwrite and iteration_complete(paths, it, finder_params):
        log.info('iteration %s already complete; skipping (use --overwrite to rerun)', it)
        summary = {'status': 'skipped'}
        if cfg.consensus:
            summary['consensus'] = run_consensus(paths, vol_frac=cfg.consensus_vol_frac, v_cut=cfg.consensus_v_cut,
                                                 threads=cfg.threads, log=log)
        return summary
    if not cfg.overwrite and iteration_complete(paths, it):
        raise FileExistsError(f'{paths.summary(it)} was produced with other void parameters: use --overwrite '
                              'or another --voids-subdir')

    summary = {'status': 'running', 'iteration': it, 'seed': cfg.seed, 'started_at': _now(),
               'config': to_jsonable(cfg), 'finder_params': finder_params, 'host': host_info(),
               'cgal': _core.CGAL_VERSION, 'membership': cfg.membership}
    write_json(paths.summary(it), summary)
    t0 = time.monotonic()
    with ResourceMonitor(log, cfg.log_interval) as mon:
        try:
            cat = loader(cfg, log, mon)
            summary.update(cat.meta)
            summary['n_points'] = cat.size
            run_paths = RunPaths(cfg.run_dir)
            points_path = run_paths.points(it)
            if cfg.save_points and not valid_npy(points_path, cat.size, points_dtype(list(cat.columns))):
                with mon.stage('guardar_puntos'):
                    save_points(cat, points_path)
                log.info('points saved to %s', points_path)
            all_voids, clean, members, stats = find_voids(cat, cfg, it, log, mon, membership=cfg.membership)
            with mon.stage('guardar_voids'):
                write_structured(paths.catalog(it), {n: all_voids[n] for n in CATALOG_DTYPE.names}, CATALOG_DTYPE)
                write_structured(paths.clean(it), {n: clean[n] for n in CLEAN_DTYPE.names}, CLEAN_DTYPE)
                if members is not None:
                    write_structured(paths.membership(it), members, MEMBERSHIP_DTYPE)
            del members, cat
            summary.update(status='complete', completed_at=_now(), stats=stats,
                           n_catalog=int(len(all_voids)), n_clean=int(len(clean)),
                           elapsed_seconds=round(time.monotonic() - t0, 2),
                           timings_seconds={k: round(v, 3) for k, v in mon.timings.items()},
                           peak_rss_gib=round(peak_rss_bytes() / 2**30, 2),
                           catalog=str(paths.catalog(it)), clean=str(paths.clean(it)),
                           membership_file=str(paths.membership(it)) if cfg.membership else None,
                           points=str(points_path) if points_path.exists() else None)
            write_json(paths.summary(it), summary)
            log.info('iteration %s completed in %.1f s | peak memory %.2f gib',
                     it, summary['elapsed_seconds'], summary['peak_rss_gib'])
        except BaseException as exc:
            log.exception('iteration %s failed', it)
            summary.update(status='failed', failed_at=_now(), error=repr(exc))
            write_json(paths.summary(it), summary)
            raise
        if cfg.consensus:
            with mon.stage('consenso'):
                summary['consensus'] = run_consensus(paths, vol_frac=cfg.consensus_vol_frac,
                                                     v_cut=cfg.consensus_v_cut, threads=cfg.threads, log=log)
    return summary


def run_cutsky_voids(cfg: VoidCutskyConfig) -> dict:
    return run_void_iteration(cfg, load_cutsky)


def run_cubic_voids(cfg: VoidCubicConfig) -> dict:
    return run_void_iteration(cfg, load_cubic)

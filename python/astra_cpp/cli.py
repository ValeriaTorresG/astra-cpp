"""command line: `python -m astra_cpp {cutsky,cubic,aggregate,info} ...`."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import (DEFAULT_CUBIC_BOX, DEFAULT_CUTSKY_RANDOM_GLOB, DEFAULT_CUTSKY_REAL, DEFAULT_OUTPUT,
                     CubicConfig, CutskyConfig, default_threads)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument('--iteration', type=int, default=0, help='random iteration (determines the seed)')
    p.add_argument('--seed-offset', type=int, default=0, help='seed = seed-offset + iteration')
    p.add_argument('--threads', type=int, default=default_threads())
    p.add_argument('--triangulation-threads', type=int, default=0,
                   help='cgal insertion threads (0: min(threads, 64))')
    p.add_argument('--lock-grid-cells', type=int, default=0,
                   help='lock grid resolution per axis (0: automatic ∝ n^(1/3))')
    p.add_argument('--r-lower', type=float, default=-0.25)
    p.add_argument('--r-med', type=float, default=0.25)
    p.add_argument('--r-upper', type=float, default=0.65)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--no-save-points', dest='save_points', action='store_false',
                   help='do not save positions (classification only)')
    p.add_argument('--no-aggregate', dest='aggregate', action='store_false',
                   help='do not recompute probabilities at the end')
    p.add_argument('--overwrite', action='store_true', help='rerun even if the iteration is complete')
    p.add_argument('--log-interval', type=float, default=30.0, help='seconds between resource logs')


def _params(args, names) -> dict:
    return {k: getattr(args, k) for k in names}


COMMON = ('iteration', 'seed_offset', 'threads', 'triangulation_threads', 'lock_grid_cells', 'r_lower', 'r_med', 'r_upper',
          'output_dir', 'save_points', 'aggregate', 'overwrite', 'log_interval')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m astra_cpp', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('cutsky', help='one iteration on a cut-sky mock + random pool')
    _common(p)
    p.add_argument('--real-file', type=Path, default=DEFAULT_CUTSKY_REAL)
    p.add_argument('--random-glob', default=DEFAULT_CUTSKY_RANDOM_GLOB)
    p.add_argument('--redshift', choices=('Z', 'Z_COSMO'), default='Z')
    p.add_argument('--max-real-rows', type=int, default=None, help='first n real rows (tests)')
    p.add_argument('--footprint', default='IN_Y5',
                   help='keep only real and random rows where this column is nonzero (default: IN_Y5)')
    p.add_argument('--no-footprint', dest='footprint', action='store_const', const=None,
                   help='use the full catalogs, without a footprint cut')

    p = sub.add_parser('cubic', help='one iteration on a cubic box + uniform randoms')
    _common(p)
    p.add_argument('--box-file', type=Path, default=DEFAULT_CUBIC_BOX)
    p.add_argument('--box-size', type=float, default=2000.0)
    p.add_argument('--random-factor', type=float, default=1.0)
    p.add_argument('--no-periodic', dest='periodic', action='store_false',
                   help='triangulate the box without periodic conditions')
    p.add_argument('--subbox', type=float, default=None, help='side length of a nonperiodic subbox (tests)')
    p.add_argument('--ghost-margin', type=float, default=0.0,
                   help='periodic copy layer width (0: 8 mean spacings)')

    p = sub.add_parser('aggregate', help='probabilities from completed iterations')
    p.add_argument('run_dir', type=Path)
    p.add_argument('--threads', type=int, default=default_threads())

    sub.add_parser('info', help='cgal version and compiled core capabilities')

    args = parser.parse_args(argv)
    if args.command == 'info':
        from . import _core
        print(f'CGAL {_core.CGAL_VERSION} | triangulación paralela (TBB): '
              f'{_core.HAS_PARALLEL_TRIANGULATION} | hilos OpenMP: {_core.max_threads()}')
        return 0
    if args.command == 'aggregate':
        from .monitor import setup_logger
        from .outputs import RunPaths, aggregate
        return 0 if aggregate(RunPaths(args.run_dir), args.threads, setup_logger()) else 1

    from .pipeline import run_cubic, run_cutsky
    if args.command == 'cutsky':
        cfg = CutskyConfig(**_params(args, COMMON + ('real_file', 'random_glob', 'redshift', 'max_real_rows', 'footprint')))
        run_cutsky(cfg)
    else:
        cfg = CubicConfig(**_params(args, COMMON + ('box_file', 'box_size', 'random_factor', 'periodic', 'ghost_margin', 'subbox')))
        run_cubic(cfg)
    return 0

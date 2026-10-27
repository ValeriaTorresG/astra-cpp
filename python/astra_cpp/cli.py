"""Línea de comandos: `python -m astra_cpp {cutsky,cubic,aggregate,info} ...`."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import (DEFAULT_CUBIC_BOX, DEFAULT_CUTSKY_RANDOM_GLOB, DEFAULT_CUTSKY_REAL, DEFAULT_OUTPUT,
                     CubicConfig, CutskyConfig, default_threads)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument('--iteration', type=int, default=0, help='iteración de randoms (define la semilla)')
    p.add_argument('--seed-offset', type=int, default=0, help='semilla = seed-offset + iteration')
    p.add_argument('--threads', type=int, default=default_threads())
    p.add_argument('--triangulation-threads', type=int, default=0,
                   help='hilos de la inserción en CGAL (0: min(threads, 64))')
    p.add_argument('--lock-grid-cells', type=int, default=0,
                   help='resolución por eje de la grilla de locks (0: automática ∝ N^(1/3))')
    p.add_argument('--r-lower', type=float, default=-0.25)
    p.add_argument('--r-med', type=float, default=0.25)
    p.add_argument('--r-upper', type=float, default=0.65)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--no-save-points', dest='save_points', action='store_false',
                   help='no guardar posiciones (solo la clasificación)')
    p.add_argument('--no-aggregate', dest='aggregate', action='store_false',
                   help='no recalcular las probabilidades al terminar')
    p.add_argument('--overwrite', action='store_true', help='repetir aunque la iteración ya esté completa')
    p.add_argument('--log-interval', type=float, default=30.0, help='segundos entre registros de recursos')


def _params(args, names) -> dict:
    return {k: getattr(args, k) for k in names}


COMMON = ('iteration', 'seed_offset', 'threads', 'triangulation_threads', 'lock_grid_cells', 'r_lower', 'r_med', 'r_upper',
          'output_dir', 'save_points', 'aggregate', 'overwrite', 'log_interval')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m astra_cpp', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('cutsky', help='una iteración sobre un mock cut-sky + pool de randoms')
    _common(p)
    p.add_argument('--real-file', type=Path, default=DEFAULT_CUTSKY_REAL)
    p.add_argument('--random-glob', default=DEFAULT_CUTSKY_RANDOM_GLOB)
    p.add_argument('--redshift', choices=('Z', 'Z_COSMO'), default='Z')
    p.add_argument('--max-real-rows', type=int, default=None, help='primeras N filas reales (pruebas)')

    p = sub.add_parser('cubic', help='una iteración sobre una caja cúbica + randoms uniformes')
    _common(p)
    p.add_argument('--box-file', type=Path, default=DEFAULT_CUBIC_BOX)
    p.add_argument('--box-size', type=float, default=2000.0)
    p.add_argument('--random-factor', type=float, default=1.0)
    p.add_argument('--no-periodic', dest='periodic', action='store_false',
                   help='triangular la caja sin condiciones periódicas')
    p.add_argument('--subbox', type=float, default=None, help='lado de una sub-caja no periódica (pruebas)')
    p.add_argument('--ghost-margin', type=float, default=0.0,
                   help='ancho de la capa de copias periódicas (0: 8 espaciados medios)')

    p = sub.add_parser('aggregate', help='probabilidades a partir de las iteraciones completas')
    p.add_argument('run_dir', type=Path)
    p.add_argument('--threads', type=int, default=default_threads())

    sub.add_parser('info', help='versión de CGAL y capacidades del núcleo compilado')

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
        cfg = CutskyConfig(**_params(args, COMMON + ('real_file', 'random_glob', 'redshift', 'max_real_rows')))
        run_cutsky(cfg)
    else:
        cfg = CubicConfig(**_params(args, COMMON + ('box_file', 'box_size', 'random_factor', 'periodic', 'ghost_margin', 'subbox')))
        run_cubic(cfg)
    return 0

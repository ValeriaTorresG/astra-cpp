"""command line: `python -m astra_cpp.voids {cutsky,cubic,consensus} ...`.

astra void finder (forero/astra-desi, branch void-finder) on the same catalogs
as the classification: cutsky/cubic take the same catalog options and seed, and
write to <classification run dir>/voids/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..cli import CUBIC, CUTSKY, RUN, _params, add_cubic_args, add_cutsky_args, add_run_args
from ..config import default_threads
from . import config as C


def _void_args(p: argparse.ArgumentParser, cutsky: bool) -> None:
    g = p.add_argument_group('void finder')
    g.add_argument('--r-threshold', type=float, default=C.DEFAULT_R_THRESHOLD,
                   help='points with r <= this value enter the watershed')
    g.add_argument('--min-members', type=int, default=C.DEFAULT_MIN_MEMBERS,
                   help='minimum group size, data + randoms')
    g.add_argument('--h', type=float, default=0.0,
                   help='catalog coordinates = positions * h (0: planck18 h for mpc, 1 for mpc/h)')
    g.add_argument('--no-membership', dest='membership', action='store_false',
                   help='do not write the per-point membership file')
    g.add_argument('--voids-subdir', default='voids', help='output subdirectory of the run directory')
    g.add_argument('--consensus', action='store_true',
                   help='rebuild the consensus over every completed iteration at the end')
    g.add_argument('--consensus-vol-frac', type=float, default=C.DEFAULT_CONSENSUS_VOL_FRAC)
    g.add_argument('--consensus-v-cut', type=float, default=C.DEFAULT_CONSENSUS_V_CUT)
    if cutsky:
        g.add_argument('--no-mask', dest='mask', action='store_false',
                       help='skip the angular/radial selection (no border groups)')
        g.add_argument('--healpix-nside', type=int, default=C.DEFAULT_NSIDE)
        g.add_argument('--min-randoms-per-pixel', type=int, default=C.DEFAULT_MIN_RANDOMS_PER_PIXEL)
        g.add_argument('--min-randoms-per-radial-bin', type=int, default=C.DEFAULT_MIN_RANDOMS_PER_RADIAL_BIN)
        g.add_argument('--radial-bin-width', type=float, default=C.DEFAULT_RADIAL_BIN_WIDTH,
                       help='radial bin width in mpc')
        g.add_argument('--angular-sample-step', type=float, default=0.0,
                       help='radians between samples along internal edges (0: half a healpix pixel)')
    else:
        g.add_argument('--boundary-buffer', type=float, default=-1.0,
                       help='nonperiodic box: groups within this distance of a face are border '
                            '(< 0: mean random spacing, 0: none)')


VOID_COMMON = ('r_threshold', 'min_members', 'h', 'membership', 'voids_subdir', 'consensus', 'consensus_vol_frac',
               'consensus_v_cut')
VOID_CUTSKY = ('mask', 'healpix_nside', 'min_randoms_per_pixel', 'min_randoms_per_radial_bin', 'radial_bin_width',
               'angular_sample_step')
VOID_CUBIC = ('boundary_buffer',)


def _iterations(tokens: list[str] | None) -> list[int] | None:
    """'0-9', '0:20:2', '3' -> iterations."""
    if not tokens:
        return None
    out = []
    for token in tokens:
        for text in token.split(','):
            text = text.strip()
            if not text:
                continue
            if ':' in text:
                parts = [int(x) if x else None for x in text.split(':')]
                out.extend(range(parts[0] or 0, parts[1], parts[2] if len(parts) > 2 and parts[2] else 1))
            elif '-' in text[1:]:
                a, b = text.split('-', 1)
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(text))
    return list(dict.fromkeys(out))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m astra_cpp.voids', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('cutsky', help='voids of one iteration on a cut-sky mock + random pool')
    add_run_args(p)
    add_cutsky_args(p)
    _void_args(p, cutsky=True)

    p = sub.add_parser('cubic', help='voids of one iteration on a cubic box + uniform randoms')
    add_run_args(p)
    add_cubic_args(p)
    _void_args(p, cutsky=False)

    p = sub.add_parser('consensus', help='consensus catalog over completed void iterations')
    p.add_argument('run_dir', type=Path, help='classification run directory (contains voids/)')
    p.add_argument('--voids-subdir', default='voids')
    p.add_argument('--iterations', nargs='+', default=None,
                   help="iterations, e.g. '0-9' or '0:20:2' (default: every completed iteration)")
    p.add_argument('--vol-frac', type=float, default=C.DEFAULT_CONSENSUS_VOL_FRAC)
    p.add_argument('--v-cut', type=float, default=C.DEFAULT_CONSENSUS_V_CUT)
    p.add_argument('--keep-all', action='store_true', help='skip the support cut')
    p.add_argument('--threads', type=int, default=default_threads())

    args = parser.parse_args(argv)
    if args.command == 'consensus':
        from ..monitor import setup_logger
        from .consensus import run_consensus
        from .outputs import VoidPaths
        if not 0.5 <= args.vol_frac <= 1.0 or not 0.0 <= args.v_cut <= 1.0:
            parser.error('--vol-frac must lie within [0.5, 1] and --v-cut within [0, 1]')
        summary = run_consensus(VoidPaths(args.run_dir / args.voids_subdir), _iterations(args.iterations),
                                vol_frac=args.vol_frac, v_cut=args.v_cut, keep_all=args.keep_all,
                                threads=args.threads, log=setup_logger(name='astra_cpp.voids'))
        return 0 if summary else 1

    from .pipeline import run_cubic_voids, run_cutsky_voids
    if args.command == 'cutsky':
        run_cutsky_voids(C.VoidCutskyConfig(**_params(args, RUN + CUTSKY + VOID_COMMON + VOID_CUTSKY)))
    else:
        run_cubic_voids(C.VoidCubicConfig(**_params(args, RUN + CUBIC + VOID_COMMON + VOID_CUBIC)))
    return 0

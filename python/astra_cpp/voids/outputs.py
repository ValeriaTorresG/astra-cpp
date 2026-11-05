"""void finder outputs.

structure of `run_dir/voids` (run_dir is the classification run directory, so
the membership rows align with `run_dir/points/iter_XXX.npy`):
    catalog/iter_XXX.npy      one row per group after the mask (all.fits in astra-desi)
    clean/iter_XXX.npy        BORDER == False subset, without the border columns
    membership/iter_XXX.npy   one row per point (data then randoms, as points/)
    summary/iter_XXX.json     parameters, counts, timings, and memory
    logs/iter_XXX.log
    consensus/voids_nNNN[_all].npy/.json   consensus over NNN iterations
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..outputs import read_json, valid_npy

EIGVAL_FIELDS = tuple(f'EIGVAL_{a}' for a in range(1, 4))
EIGVEC_FIELDS = tuple(f'EIGVEC_{a}_{c}' for a in range(1, 4) for c in 'XYZ')
SHAPE_FIELDS = EIGVAL_FIELDS + EIGVEC_FIELDS

_CLEAN_FIELDS = ([('VOID_ID', '<i8'), ('GROUP_ID', '<i4'), ('XCART', '<f8'), ('YCART', '<f8'), ('ZCART', '<f8'),
                  ('R_EFF', '<f8'), ('ELLIP', '<f8')] + [(f, '<f8') for f in SHAPE_FIELDS]
                 + [('N_MEMBERS', '<i8'), ('N_DATA', '<i8'), ('N_RANDOM', '<i8')])
CLEAN_DTYPE = np.dtype(_CLEAN_FIELDS)
# BORDER_FLAGS: bits of astra_cpp._core.BORDER_* (angular/radial member or edge, box face)
CATALOG_DTYPE = np.dtype(_CLEAN_FIELDS + [('BORDER', '?'), ('BORDER_FLAGS', 'u1')])
MEMBERSHIP_DTYPE = np.dtype([('TARGETID', '<i8'), ('ISDATA', '?'), ('R', '<f4'), ('SELECTED', '?'),
                             ('GROUP_ID_PREMASK', '<i4'), ('GROUP_ID', '<i4'), ('VOID_ID', '<i8'),
                             ('PRUNED', '?'), ('BORDER', '?')])
CONSENSUS_DTYPE = np.dtype([('VOID_ID', '<i8'), ('SRC_ITER', '<i4'), ('X', '<f8'), ('Y', '<f8'), ('Z', '<f8'),
                            ('R_EFF', '<f8'), ('ELLIP', '<f8')] + [(f, '<f8') for f in SHAPE_FIELDS]
                           + [('V', '<i4'), ('FRAC_V', '<f8'), ('N_ABSORBED', '<i4')])


class VoidPaths:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _iter(self, kind: str, iteration: int, ext: str) -> Path:
        return self.root / kind / f'iter_{iteration:03d}.{ext}'

    def catalog(self, iteration: int) -> Path:
        return self._iter('catalog', iteration, 'npy')

    def clean(self, iteration: int) -> Path:
        return self._iter('clean', iteration, 'npy')

    def membership(self, iteration: int) -> Path:
        return self._iter('membership', iteration, 'npy')

    def summary(self, iteration: int) -> Path:
        return self._iter('summary', iteration, 'json')

    def log(self, iteration: int) -> Path:
        return self._iter('logs', iteration, 'log')

    def consensus(self, n_runs: int, keep_all: bool = False, ext: str = 'npy') -> Path:
        return self.root / 'consensus' / f'voids_n{n_runs:03d}{"_all" if keep_all else ""}.{ext}'

    def make_dirs(self) -> None:
        for kind in ('catalog', 'clean', 'membership', 'summary', 'logs', 'consensus'):
            (self.root / kind).mkdir(parents=True, exist_ok=True)

    def summaries(self) -> list[tuple[int, Path]]:
        return [(int(p.stem.split('_')[1]), p) for p in sorted((self.root / 'summary').glob('iter_*.json'))]


def iteration_complete(paths: VoidPaths, iteration: int, finder_params: dict | None = None) -> bool:
    """complete iteration with valid outputs (and the same finder parameters, if given)."""
    summary = read_json(paths.summary(iteration))
    if not summary or summary.get('status') != 'complete':
        return False
    if finder_params is not None and summary.get('finder_params') != finder_params:
        return False
    if not valid_npy(paths.catalog(iteration), summary['n_catalog'], CATALOG_DTYPE):
        return False
    if not valid_npy(paths.clean(iteration), summary['n_clean'], CLEAN_DTYPE):
        return False
    if summary.get('membership') and not valid_npy(paths.membership(iteration), summary['n_points'], MEMBERSHIP_DTYPE):
        return False
    return True


def completed_iterations(paths: VoidPaths) -> list[int]:
    return [it for it, _ in paths.summaries() if iteration_complete(paths, it)]

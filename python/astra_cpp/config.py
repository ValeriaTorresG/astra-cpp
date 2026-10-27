"""default paths, astra thresholds, and configuration for one iteration."""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

ABACUS_ROOT = Path('/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit')
CUTSKY_DIR = ABACUS_ROOT / 'CutSky' / 'BGS' / 'v0' / 'z0.200'
DEFAULT_CUTSKY_REAL = CUTSKY_DIR / 'cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits'
DEFAULT_CUTSKY_RANDOM_GLOB = str(CUTSKY_DIR / 'random_S*_1X.fits')
DEFAULT_CUBIC_BOX = (ABACUS_ROOT / 'CubicBox' / 'BGS' / 'v0' / 'z0.200' /
                     'AbacusSummit_base_c000_ph000' / 'BGS_box_ph000.fits')
DEFAULT_OUTPUT = Path('/pscratch/sd/v/vtorresg/astra-cpp-runs')

# randoms from the same seed file have targetid = -(seed * factor + row + 1)
RANDOM_ID_FACTOR = 10_000_000_000
WEBTYPES = ('void', 'sheet', 'filament', 'knot')
MAX_TRIANGULATION_THREADS = 64
# default periodic copy layer width in mean point spacings. the largest empty
# sphere among uniform randoms is ~2 spacings, so 8 leaves a margin; the c++
# verification detects if that margin is insufficient.
GHOST_SPACINGS = 8.0


def default_threads() -> int:
    return len(os.sched_getaffinity(0))


@dataclass
class AstraParams:
    """parameters shared by all catalogs."""
    iteration: int = 0
    seed_offset: int = 0          # iteration seed = seed_offset + iteration
    r_lower: float = -0.25
    r_med: float = 0.25
    r_upper: float = 0.65
    threads: int = field(default_factory=default_threads)
    triangulation_threads: int = 0  # <= 0: min(threads, MAX_TRIANGULATION_THREADS)
    lock_grid_cells: int = 0        # <= 0: automatic (∝ n^(1/3))
    output_dir: Path = DEFAULT_OUTPUT
    save_points: bool = True
    aggregate: bool = True
    overwrite: bool = False
    log_interval: float = 30.0

    @property
    def tri_threads(self) -> int:
        """threads for parallel cgal insertion. beyond a few dozen threads,
        lock contention grows and may cause livelock, so the default is capped
        even when other stages use all available threads."""
        if self.triangulation_threads > 0:
            return self.triangulation_threads
        return min(self.threads, MAX_TRIANGULATION_THREADS)

    @property
    def seed(self) -> int:
        return self.seed_offset + self.iteration

    def validate(self) -> None:
        if self.iteration < 0:
            raise ValueError('iteration must be >= 0')
        if not (self.r_lower < self.r_med < self.r_upper):
            raise ValueError('r_lower < r_med < r_upper is required')
        if self.threads <= 0:
            raise ValueError('threads must be positive')


@dataclass
class CutskyConfig(AstraParams):
    real_file: Path = DEFAULT_CUTSKY_REAL
    random_glob: str = DEFAULT_CUTSKY_RANDOM_GLOB
    redshift: str = 'Z'                    # 'Z' (with rsd) or 'Z_COSMO'
    max_real_rows: int | None = None       # first n real rows (tests)

    @property
    def run_dir(self) -> Path:
        return self.output_dir / 'cutsky' / self.real_file.stem / self.redshift.lower()


@dataclass
class CubicConfig(AstraParams):
    box_file: Path = DEFAULT_CUBIC_BOX
    box_size: float = 2000.0               # mpc/h (abacussummit base)
    random_factor: float = 1.0             # n_random = random_factor * n_data
    periodic: bool = True                  # periodic conditions (ghost copies)
    ghost_margin: float = 0.0              # periodic layer; <= 0: ghost_spacings * mean spacing
    subbox: float | None = None            # side length of a nonperiodic subbox (tests)

    @property
    def run_dir(self) -> Path:
        tag = self.box_file.stem + (f'_subbox{self.subbox:g}' if self.subbox else '')
        return self.output_dir / 'cubic' / tag


def to_jsonable(cfg) -> dict:
    out = {}
    for key, value in dataclasses.asdict(cfg).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out

"""Rutas por defecto, umbrales de ASTRA y configuración de una iteración."""
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

# Los randoms de un mismo archivo semilla tienen TARGETID = -(seed * FACTOR + fila + 1)
RANDOM_ID_FACTOR = 10_000_000_000
WEBTYPES = ('void', 'sheet', 'filament', 'knot')
MAX_TRIANGULATION_THREADS = 64
# Ancho por defecto de la capa de copias periódicas, en espaciados medios entre
# puntos. La esfera vacía más grande entre randoms uniformes es ~2 espaciados, así
# que 8 deja margen; si no alcanzara, la verificación en C++ lo detecta.
GHOST_SPACINGS = 8.0


def default_threads() -> int:
    return len(os.sched_getaffinity(0))


@dataclass
class AstraParams:
    """Parámetros comunes a cualquier catálogo."""
    iteration: int = 0
    seed_offset: int = 0          # semilla de la iteración = seed_offset + iteration
    r_lower: float = -0.25
    r_med: float = 0.25
    r_upper: float = 0.65
    threads: int = field(default_factory=default_threads)
    triangulation_threads: int = 0  # <= 0: min(threads, MAX_TRIANGULATION_THREADS)
    lock_grid_cells: int = 0        # <= 0: automática (∝ N^(1/3))
    output_dir: Path = DEFAULT_OUTPUT
    save_points: bool = True
    aggregate: bool = True
    overwrite: bool = False
    log_interval: float = 30.0

    @property
    def tri_threads(self) -> int:
        """Hilos de la inserción paralela en CGAL. Más allá de unas decenas la
        contención de locks crece y puede degenerar en livelock, así que por
        defecto se limita aunque el resto de etapas use todos los hilos."""
        if self.triangulation_threads > 0:
            return self.triangulation_threads
        return min(self.threads, MAX_TRIANGULATION_THREADS)

    @property
    def seed(self) -> int:
        return self.seed_offset + self.iteration

    def validate(self) -> None:
        if self.iteration < 0:
            raise ValueError('iteration debe ser >= 0')
        if not (self.r_lower < self.r_med < self.r_upper):
            raise ValueError('se requiere r_lower < r_med < r_upper')
        if self.threads <= 0:
            raise ValueError('threads debe ser positivo')


@dataclass
class CutskyConfig(AstraParams):
    real_file: Path = DEFAULT_CUTSKY_REAL
    random_glob: str = DEFAULT_CUTSKY_RANDOM_GLOB
    redshift: str = 'Z'                    # 'Z' (con RSD) o 'Z_COSMO'
    max_real_rows: int | None = None       # primeras N filas reales (pruebas)

    @property
    def run_dir(self) -> Path:
        return self.output_dir / 'cutsky' / self.real_file.stem / self.redshift.lower()


@dataclass
class CubicConfig(AstraParams):
    box_file: Path = DEFAULT_CUBIC_BOX
    box_size: float = 2000.0               # Mpc/h (AbacusSummit base)
    random_factor: float = 1.0             # n_random = random_factor * n_data
    periodic: bool = True                  # condiciones periódicas (copias fantasma)
    ghost_margin: float = 0.0              # capa periódica; <= 0: GHOST_SPACINGS * espaciado medio
    subbox: float | None = None            # lado de una sub-caja no periódica (pruebas)

    @property
    def run_dir(self) -> Path:
        tag = self.box_file.stem + (f'_subbox{self.subbox:g}' if self.subbox else '')
        return self.output_dir / 'cubic' / tag


def to_jsonable(cfg) -> dict:
    out = {}
    for key, value in dataclasses.asdict(cfg).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out

"""void finder parameters (defaults of group_finder in astra-desi, branch void-finder)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ..config import CubicConfig, CutskyConfig

DEFAULT_R_THRESHOLD = -0.25
DEFAULT_MIN_MEMBERS = 4
DEFAULT_NSIDE = 128
DEFAULT_MIN_RANDOMS_PER_PIXEL = 3
DEFAULT_MIN_RANDOMS_PER_RADIAL_BIN = 3
DEFAULT_RADIAL_BIN_WIDTH = 5.0
DEFAULT_CONSENSUS_VOL_FRAC = 0.5
DEFAULT_CONSENSUS_V_CUT = 0.5
# VOID_ID = iteration * VOID_ID_FACTOR + GROUP_ID
VOID_ID_FACTOR = 10_000_000_000
# peak memory per point of a void iteration: the classification peak (triangulation)
# plus the positions kept for the shapes, the processing order and the adjacency
PEAK_BYTES_PER_POINT = 860


@dataclass
class VoidParams:
    """parameters of the void finder, shared by cut-sky and cubic runs."""
    r_threshold: float = DEFAULT_R_THRESHOLD   # points with r <= r_threshold enter the watershed
    min_members: int = DEFAULT_MIN_MEMBERS     # minimum group size (data + randoms)
    mask: bool = True                          # cut-sky: angular/radial selection from the randoms
    healpix_nside: int = DEFAULT_NSIDE
    min_randoms_per_pixel: int = DEFAULT_MIN_RANDOMS_PER_PIXEL
    min_randoms_per_radial_bin: int = DEFAULT_MIN_RANDOMS_PER_RADIAL_BIN
    radial_bin_width: float = DEFAULT_RADIAL_BIN_WIDTH  # units of the positions (mpc for cut-sky)
    angular_sample_step: float = 0.0           # radians along internal edges; <= 0: half a pixel
    boundary_buffer: float = -1.0              # nonperiodic box: < 0 mean random spacing, 0 no border
    h: float = 0.0                             # catalog coordinates = positions * h; <= 0: automatic
    membership: bool = True                    # per-point membership file
    consensus: bool = False                    # rebuild the consensus catalog after the iteration
    consensus_vol_frac: float = DEFAULT_CONSENSUS_VOL_FRAC
    consensus_v_cut: float = DEFAULT_CONSENSUS_V_CUT
    voids_subdir: str = 'voids'                # outputs in run_dir / voids_subdir

    def validate_voids(self) -> None:
        if not -1.0 <= self.r_threshold <= 1.0:
            raise ValueError('r_threshold must lie within [-1, 1]')
        for name in ('min_members', 'healpix_nside', 'min_randoms_per_pixel', 'min_randoms_per_radial_bin'):
            if int(getattr(self, name)) < 1:
                raise ValueError(f'{name} must be positive')
        if self.healpix_nside & (self.healpix_nside - 1):
            raise ValueError('healpix_nside must be a power of 2')
        if not self.radial_bin_width > 0:
            raise ValueError('radial_bin_width must be positive')
        if not 0.5 <= self.consensus_vol_frac <= 1.0:
            raise ValueError('consensus_vol_frac must lie within [0.5, 1]')
        if not 0.0 <= self.consensus_v_cut <= 1.0:
            raise ValueError('consensus_v_cut must lie within [0, 1]')
        if not self.voids_subdir or Path(self.voids_subdir).name != self.voids_subdir:
            raise ValueError('voids_subdir must be a plain directory name')

    def sample_step(self) -> float:
        """angular step along internal edges: half the healpix pixel size by default."""
        if self.angular_sample_step > 0:
            return self.angular_sample_step
        return 0.5 * math.sqrt(4 * math.pi / (12 * self.healpix_nside ** 2))

    def finder_params(self) -> dict:
        """parameters that define the void catalogs (compared when resuming)."""
        return {'r_threshold': self.r_threshold, 'min_members': self.min_members, 'mask': self.mask,
                'healpix_nside': self.healpix_nside, 'min_randoms_per_pixel': self.min_randoms_per_pixel,
                'min_randoms_per_radial_bin': self.min_randoms_per_radial_bin,
                'radial_bin_width': self.radial_bin_width, 'angular_sample_step': self.sample_step(),
                'boundary_buffer': self.boundary_buffer, 'h': self.h}


@dataclass
class VoidCutskyConfig(CutskyConfig, VoidParams):
    def validate(self) -> None:
        super().validate()
        self.validate_voids()

    @property
    def void_dir(self) -> Path:
        return self.run_dir / self.voids_subdir


@dataclass
class VoidCubicConfig(CubicConfig, VoidParams):
    def validate(self) -> None:
        super().validate()
        self.validate_voids()

    @property
    def void_dir(self) -> Path:
        return self.run_dir / self.voids_subdir

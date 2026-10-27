"""
astra with a c++ (cgal) core: triangulation, neighbors, r, and classification.

the compiled `_core` module does the heavy work; this package orchestrates
the stages and saves the outputs.
"""
from .config import CubicConfig, CutskyConfig
from .pipeline import classify_catalog, run_cubic, run_cutsky

__all__ = ['CubicConfig', 'CutskyConfig', 'classify_catalog', 'run_cubic', 'run_cutsky']

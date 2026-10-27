"""ASTRA con el núcleo en C++ (CGAL): triangulación, vecinos, r y clasificación.

El módulo compilado `_core` hace todo el trabajo pesado; este paquete solo
orquesta las etapas y guarda los productos.
"""
from .config import CubicConfig, CutskyConfig
from .pipeline import classify_catalog, run_cubic, run_cutsky

__all__ = ['CubicConfig', 'CutskyConfig', 'classify_catalog', 'run_cubic', 'run_cutsky']

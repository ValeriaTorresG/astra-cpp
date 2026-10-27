#!/bin/bash
# Compila el núcleo C++ (CGAL + TBB + OpenMP) y deja astra_cpp/_core*.so en python/astra_cpp/.
# Uso:
#   source /pscratch/sd/v/vtorresg/venvs/triang/bin/activate
#   bash build.sh            # Release con TBB
#   ASTRA_WITH_TBB=OFF bash build.sh   # solo triangulación secuencial
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CGAL_DIR=${CGAL_DIR:-/pscratch/sd/v/vtorresg/deps/CGAL-5.6.2}
# GMP y MPFR (headers + libs) vienen del Python de NERSC; Boost 1.66 y TBB 2019 del sistema
NERSC_PY=${NERSC_PY:-/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.12/nersc-python}
BUILD_DIR=${BUILD_DIR:-$ROOT/build}
JOBS=${JOBS:-8}

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activa primero el entorno: source /pscratch/sd/v/vtorresg/venvs/triang/bin/activate" >&2
  exit 1
fi

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${CXX:-g++}" \
  -DPython_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCGAL_INCLUDE_DIR="$CGAL_DIR/include" \
  -DGMP_PREFIX="$NERSC_PY" \
  -DTBB_DIR=/usr/lib64/cmake/TBB \
  -DASTRA_WITH_TBB="${ASTRA_WITH_TBB:-ON}"
cmake --build "$BUILD_DIR" -j "$JOBS"

python -c "import sys; sys.path.insert(0, '$ROOT/python'); from astra_cpp import _core as c; \
print('Compilado astra_cpp._core | CGAL', c.CGAL_VERSION, '| triangulación paralela:', c.HAS_PARALLEL_TRIANGULATION)"

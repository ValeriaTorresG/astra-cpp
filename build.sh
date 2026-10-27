#!/bin/bash
# build the c++ core (cgal + tbb + openmp) and place astra_cpp/_core*.so in python/astra_cpp/.
# usage:
#   source /pscratch/sd/v/vtorresg/venvs/triang/bin/activate
#   bash build.sh            # release with tbb
#   ASTRA_WITH_TBB=OFF bash build.sh   # sequential triangulation only
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CGAL_DIR=${CGAL_DIR:-/pscratch/sd/v/vtorresg/deps/CGAL-5.6.2}
# gmp and mpfr (headers and libraries) come from nersc python; boost 1.66 and tbb 2019 come from the system
NERSC_PY=${NERSC_PY:-/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.12/nersc-python}
BUILD_DIR=${BUILD_DIR:-$ROOT/build}
JOBS=${JOBS:-8}

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "use: source /pscratch/sd/v/vtorresg/venvs/triang/bin/activate" >&2
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
print('compiled astra_cpp._core | CGAL', c.CGAL_VERSION, '| paralel triang:', c.HAS_PARALLEL_TRIANGULATION)"

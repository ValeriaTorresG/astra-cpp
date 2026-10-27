cmake_minimum_required(VERSION 3.18)
project(astra_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CGAL es header-only; GMP/MPFR los necesita el kernel Epick para los predicados exactos.
set(CGAL_INCLUDE_DIR "" CACHE PATH "Directorio include de CGAL")
set(GMP_PREFIX "" CACHE PATH "Prefijo con include/ y lib/ de GMP y MPFR")
option(ASTRA_WITH_TBB "Triangulación CGAL paralela con TBB" ON)

if(NOT EXISTS "${CGAL_INCLUDE_DIR}/CGAL/version.h")
  message(FATAL_ERROR "CGAL_INCLUDE_DIR no apunta a los headers de CGAL: '${CGAL_INCLUDE_DIR}'")
endif()

find_package(OpenMP REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_library(GMP_LIB gmp HINTS "${GMP_PREFIX}/lib" REQUIRED)
find_library(MPFR_LIB mpfr HINTS "${GMP_PREFIX}/lib" REQUIRED)

# ---------------------------------------------------------------- núcleo C++
add_library(astra_core STATIC
  cpp/src/common.cpp
  cpp/src/box.cpp
  cpp/src/fits.cpp
  cpp/src/cosmology.cpp
  cpp/src/random_pool.cpp
  cpp/src/delaunay.cpp
  cpp/src/classify.cpp)
target_include_directories(astra_core PUBLIC cpp/include)
target_include_directories(astra_core SYSTEM PUBLIC "${CGAL_INCLUDE_DIR}" "${GMP_PREFIX}/include")
target_compile_definitions(astra_core PUBLIC CGAL_DISABLE_ROUNDING_MATH_CHECK)
target_compile_options(astra_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(astra_core PUBLIC OpenMP::OpenMP_CXX ${MPFR_LIB} ${GMP_LIB})

if(ASTRA_WITH_TBB)
  find_package(TBB REQUIRED)
  target_compile_definitions(astra_core PUBLIC CGAL_LINKED_WITH_TBB NOMINMAX)
  target_link_libraries(astra_core PUBLIC TBB::tbb TBB::tbbmalloc)
endif()

# ---------------------------------------------------------------- módulo Python
pybind11_add_module(_core cpp/bindings/module.cpp)
target_link_libraries(_core PRIVATE astra_core)
get_filename_component(GMP_LIB_DIR "${GMP_LIB}" DIRECTORY)
set_target_properties(_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/astra_cpp"
  BUILD_RPATH "${GMP_LIB_DIR}"
  INSTALL_RPATH "${GMP_LIB_DIR}")

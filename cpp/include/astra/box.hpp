// Utilidades para cajas cúbicas periódicas.
#pragma once

#include <array>

#include "astra/common.hpp"

namespace astra {

// Lleva cada punto (xyz de tamaño 3n) a [lo, hi) por periodicidad.
void wrap_periodic(double* xyz, Index n, const std::array<double, 3>& lo, const std::array<double, 3>& hi,
                   int threads);

}  // namespace astra

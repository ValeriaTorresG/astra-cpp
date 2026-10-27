// Generación de los randoms de cada iteración de ASTRA.
#pragma once

#include <cstdint>
#include <vector>

#include "astra/common.hpp"

namespace astra {

// Muestra uniforme de `n` filas distintas (sin reemplazo) de un pool formado
// por la concatenación de varias fuentes de tamaños `sizes`, igual que el
// pooling de ASTRA (rng.choice(total, n, replace=False) sobre el pool unido).
// Devuelve, por fuente, las filas locales seleccionadas en orden creciente.
// El resultado depende solo de (sizes, n, seed).
std::vector<std::vector<Index>> sample_pool(const std::vector<Index>& sizes, Index n,
                                            std::uint64_t seed);

// n puntos uniformes en la caja [lo, hi) (xyz de tamaño 3n). Generador basado
// en contador: el resultado no depende del número de hilos.
void uniform_box(Index n, const double lo[3], const double hi[3], std::uint64_t seed,
                 double* xyz, int threads);

}  // namespace astra

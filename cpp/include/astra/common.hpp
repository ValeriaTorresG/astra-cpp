// Tipos y utilidades compartidas por todos los módulos del núcleo de ASTRA.
#pragma once

#include <cstddef>
#include <cstdint>

namespace astra {

// Índice de fila / punto. Con signo para poder usar -1 como centinela.
using Index = std::int64_t;

// Clases de la red cósmica en el mismo orden que ASTRA (WEBTYPE).
enum WebType : std::uint8_t { kVoid = 0, kSheet = 1, kFilament = 2, kKnot = 3 };
inline constexpr int kNumWebTypes = 4;

// Número de hilos efectivo: <= 0 significa "todos los que permita OpenMP".
int resolve_threads(int requested);

// true si el núcleo se compiló con TBB (triangulación CGAL paralela).
bool has_parallel_triangulation();

}  // namespace astra

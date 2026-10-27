// types and utilities shared by all astra core modules.
#pragma once

#include <cstddef>
#include <cstdint>

namespace astra {

// row / point index. signed to allow -1 as a sentinel.
using Index = std::int64_t;

// cosmic web classes in the same order as astra (webtype).
enum WebType : std::uint8_t { kVoid = 0, kSheet = 1, kFilament = 2, kKnot = 3 };
inline constexpr int kNumWebTypes = 4;

// effective thread count: <= 0 means all threads allowed by openmp.
int resolve_threads(int requested);

// true if the core was built with tbb (parallel cgal triangulation).
bool has_parallel_triangulation();

}  // namespace astra

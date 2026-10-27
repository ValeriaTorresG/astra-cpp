// generate randoms for each astra iteration.
#pragma once

#include <cstdint>
#include <vector>

#include "astra/common.hpp"

namespace astra {

// uniform sample of `n` distinct rows (without replacement) from a pool
// formed by concatenating sources of sizes `sizes`, as in astra pooling
// (rng.choice(total, n, replace=false) over the combined pool).
// return selected local rows for each source in ascending order.
// the result depends only on (sizes, n, seed).
std::vector<std::vector<Index>> sample_pool(const std::vector<Index>& sizes, Index n,
                                            std::uint64_t seed);

// n uniform points in the [lo, hi) box (xyz of size 3n). counter-based
// generator: the result does not depend on the thread count.
void uniform_box(Index n, const double lo[3], const double hi[3], std::uint64_t seed,
                 double* xyz, int threads);

}  // namespace astra

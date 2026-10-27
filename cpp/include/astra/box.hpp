// utilities for periodic cubic boxes.
#pragma once

#include <array>

#include "astra/common.hpp"

namespace astra {

// wrap each point (xyz of size 3n) into [lo, hi) periodically.
void wrap_periodic(double* xyz, Index n, const std::array<double, 3>& lo, const std::array<double, 3>& hi,
                   int threads);

}  // namespace astra

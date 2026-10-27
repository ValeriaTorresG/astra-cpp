// astra r statistic, webtype classification, and aggregation across iterations.
#pragma once

#include <cstddef>
#include <cstdint>

namespace astra {

struct RThresholds {
    float lower = -0.25f;
    float med = 0.25f;
    float upper = 0.65f;
};

// r = (ndata - nrand) / (ndata + nrand) in float32 (0 if there are no neighbors) and
//   void: r < lower | sheet: [lower, med) | filament: [med, upper) | knot: r >= upper,
// operation by operation as in implement_astra._compute_probability_components.
void classify(const std::int32_t* ndata, const std::int32_t* nrand, std::size_t n,
              const RThresholds& thresholds, float* r, std::uint8_t* webtype, int threads);

// counts[4*i + webtype[i]] += 1 (class count per object across iterations).
void accumulate_webtypes(const std::uint8_t* webtype, std::size_t n, std::uint16_t* counts,
                         int threads);

// prob[4*i + k] = counts[4*i + k] / n_iterations.
void counts_to_probabilities(const std::uint16_t* counts, std::size_t n, int n_iterations,
                             float* prob, int threads);

}  // namespace astra

// Estadístico r de ASTRA, clasificación en WEBTYPE y agregación entre iteraciones.
#pragma once

#include <cstddef>
#include <cstdint>

namespace astra {

struct RThresholds {
    float lower = -0.25f;
    float med = 0.25f;
    float upper = 0.65f;
};

// r = (NDATA - NRAND) / (NDATA + NRAND) en float32 (0 si no hay vecinos) y
//   void: r < lower | sheet: [lower, med) | filament: [med, upper) | knot: r >= upper,
// operación por operación igual que implement_astra._compute_probability_components.
void classify(const std::int32_t* ndata, const std::int32_t* nrand, std::size_t n,
              const RThresholds& thresholds, float* r, std::uint8_t* webtype, int threads);

// counts[4*i + webtype[i]] += 1 (conteo de clases por objeto entre iteraciones).
void accumulate_webtypes(const std::uint8_t* webtype, std::size_t n, std::uint16_t* counts,
                         int threads);

// prob[4*i + k] = counts[4*i + k] / n_iterations.
void counts_to_probabilities(const std::uint16_t* counts, std::size_t n, int n_iterations,
                             float* prob, int threads);

}  // namespace astra

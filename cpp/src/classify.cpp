#include "astra/classify.hpp"

#include "astra/common.hpp"

namespace astra {

void classify(const std::int32_t* ndata, const std::int32_t* nrand, std::size_t n, const RThresholds& t,
              float* r, std::uint8_t* webtype, int threads) {
#pragma omp parallel for num_threads(resolve_threads(threads)) schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        // Mismas operaciones float32 que NumPy: (nd - nr) / (nd + nr).
        const float nd = float(ndata[i]), nr = float(nrand[i]);
        const float denom = nd + nr;
        const float ratio = denom > 0.0f ? (nd - nr) / denom : 0.0f;
        std::uint8_t c = kVoid;
        if (ratio >= t.upper) c = kKnot;
        else if (ratio >= t.med) c = kFilament;
        else if (ratio >= t.lower) c = kSheet;
        r[i] = ratio;
        webtype[i] = c;
    }
}

void accumulate_webtypes(const std::uint8_t* webtype, std::size_t n, std::uint16_t* counts, int threads) {
#pragma omp parallel for num_threads(resolve_threads(threads)) schedule(static)
    for (std::size_t i = 0; i < n; ++i)
        if (webtype[i] < kNumWebTypes) ++counts[kNumWebTypes * i + webtype[i]];
}

void counts_to_probabilities(const std::uint16_t* counts, std::size_t n, int n_iterations, float* prob,
                             int threads) {
    const std::size_t m = kNumWebTypes * n;
#pragma omp parallel for num_threads(resolve_threads(threads)) schedule(static)
    for (std::size_t i = 0; i < m; ++i) prob[i] = float(double(counts[i]) / double(n_iterations));
}

}  // namespace astra

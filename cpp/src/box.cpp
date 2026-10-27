#include "astra/box.hpp"

#include <cmath>

namespace astra {

void wrap_periodic(double* xyz, Index n, const std::array<double, 3>& lo, const std::array<double, 3>& hi,
                   int threads) {
    const double L[3] = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
#pragma omp parallel for num_threads(resolve_threads(threads)) schedule(static)
    for (Index i = 0; i < n; ++i) {
        for (int k = 0; k < 3; ++k) {
            double& x = xyz[3 * i + k];
            if (x >= lo[k] && x < hi[k]) continue;
            x = lo[k] + (x - lo[k]) - L[k] * std::floor((x - lo[k]) / L[k]);
            if (x >= hi[k]) x = lo[k];  // redondeo justo en el borde superior
        }
    }
}

}  // namespace astra

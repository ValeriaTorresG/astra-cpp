#include "astra/random_pool.hpp"

#include <numeric>
#include <stdexcept>

namespace astra {
namespace {

inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// xoshiro256** (Blackman & Vigna), sembrado con splitmix64.
class Xoshiro256 {
public:
    explicit Xoshiro256(std::uint64_t seed) {
        for (auto& w : s_) w = seed = splitmix64(seed);
    }
    std::uint64_t next() {
        const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0]; s_[3] ^= s_[1]; s_[1] ^= s_[2]; s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }
    // Entero uniforme en [0, bound) sin sesgo (método de Lemire).
    std::uint64_t below(std::uint64_t bound) {
        __uint128_t m = __uint128_t(next()) * bound;
        auto low = std::uint64_t(m);
        if (low < bound) {
            const std::uint64_t threshold = -bound % bound;
            while (low < threshold) {
                m = __uint128_t(next()) * bound;
                low = std::uint64_t(m);
            }
        }
        return std::uint64_t(m >> 64);
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    std::uint64_t s_[4];
};

}  // namespace

std::vector<std::vector<Index>> sample_pool(const std::vector<Index>& sizes, Index n, std::uint64_t seed) {
    for (Index s : sizes)
        if (s < 0) throw std::invalid_argument("tamaños de fuente negativos");
    const Index total = std::accumulate(sizes.begin(), sizes.end(), Index(0));
    if (n < 0 || n > total) throw std::invalid_argument("el pool de randoms tiene menos filas que las pedidas");

    // Sorteo con rechazo sobre un bitset del pool completo (1 bit por fila; ~240 MB
    // para 1.9e9 filas). Si n > total/2 se sortea el complemento, que es más
    // barato y da la misma distribución.
    const bool complement = n > total / 2;
    const Index k = complement ? total - n : n;
    std::vector<std::uint64_t> bits(std::size_t(total + 63) / 64, 0);
    Xoshiro256 rng(seed);
    for (Index drawn = 0; drawn < k;) {
        const std::uint64_t r = rng.below(std::uint64_t(total));
        std::uint64_t& w = bits[r >> 6];
        const std::uint64_t mask = std::uint64_t(1) << (r & 63);
        if (!(w & mask)) { w |= mask; ++drawn; }
    }

    std::vector<std::vector<Index>> out(sizes.size());
    Index start = 0;
    for (std::size_t s = 0; s < sizes.size(); ++s) {
        const Index end = start + sizes[s];
        auto& rows = out[s];
        for (Index g = start; g < end;) {
            // Recorre palabra a palabra, limitada al rango [g, end) de esta fuente.
            const Index word = g >> 6;
            std::uint64_t w = bits[std::size_t(word)];
            if (complement) w = ~w;
            const Index word_lo = word << 6;
            w &= ~std::uint64_t(0) << (g - word_lo);
            if (end - word_lo < 64) w &= (std::uint64_t(1) << (end - word_lo)) - 1;
            while (w) {
                rows.push_back(word_lo + __builtin_ctzll(w) - start);
                w &= w - 1;
            }
            g = word_lo + 64;
        }
        start = end;
    }
    return out;
}

void uniform_box(Index n, const double lo[3], const double hi[3], std::uint64_t seed, double* xyz, int threads) {
    const std::uint64_t key = splitmix64(seed ^ 0xA5712A0C0FFEEULL);
    const double span[3] = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
#pragma omp parallel for num_threads(resolve_threads(threads)) schedule(static)
    for (Index i = 0; i < n; ++i) {
        for (int k = 0; k < 3; ++k) {
            const std::uint64_t h = splitmix64(key + std::uint64_t(3 * i + k) * 0xD1B54A32D192ED03ULL);
            const double u = double(h >> 11) * 0x1.0p-53;  // [0, 1)
            xyz[3 * i + k] = lo[k] + span[k] * u;
        }
    }
}

}  // namespace astra

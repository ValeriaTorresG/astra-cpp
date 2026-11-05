#include "astra/voids.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

#include "astra/healpix.hpp"

namespace astra {
namespace {

constexpr std::int32_t kNoGroup = -1;
constexpr double kTiny = std::numeric_limits<double>::min();  // np.finfo(float64).tiny
constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kPi = 3.14159265358979323846;

// contiguous block [lo, hi) of [0, n) for thread t of nt (static schedule).
inline void block_range(Index n, int t, int nt, Index& lo, Index& hi) {
    lo = n * t / nt;
    hi = n * (t + 1) / nt;
}

inline double norm3(const double* p) { return std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]); }

inline bool finite3(const double* p) {
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

// lock-free union-find over int32 ids. parents always point to smaller ids,
// so concurrent unions cannot create cycles.
class AtomicUnionFind {
public:
    AtomicUnionFind(Index n, int threads) : parent_(std::size_t(n)) {
#pragma omp parallel for num_threads(threads) schedule(static)
        for (Index i = 0; i < n; ++i) parent_[std::size_t(i)] = std::int32_t(i);
    }

    std::int32_t find(std::int32_t x) {
        while (true) {
            std::int32_t p = __atomic_load_n(&parent_[std::size_t(x)], __ATOMIC_ACQUIRE);
            if (p == x) return x;
            const std::int32_t gp = __atomic_load_n(&parent_[std::size_t(p)], __ATOMIC_ACQUIRE);
            if (p != gp)  // path halving
                __atomic_compare_exchange_n(&parent_[std::size_t(x)], &p, gp, false, __ATOMIC_RELEASE,
                                            __ATOMIC_RELAXED);
            x = gp;
        }
    }

    void unite(std::int32_t a, std::int32_t b) {
        while (true) {
            a = find(a);
            b = find(b);
            if (a == b) return;
            if (a < b) std::swap(a, b);  // link the larger root to the smaller one
            std::int32_t expected = a;
            if (__atomic_compare_exchange_n(&parent_[std::size_t(a)], &expected, b, false, __ATOMIC_ACQ_REL,
                                            __ATOMIC_RELAXED))
                return;
        }
    }

private:
    std::vector<std::int32_t> parent_;
};

// eigen decomposition of a symmetric 3x3 matrix by cyclic jacobi rotations
// (accurate to a few ulp of the largest eigenvalue). columns of v are the
// eigenvectors of w.
void eigen_sym3(double a[3][3], double w[3], double v[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) v[i][j] = i == j ? 1.0 : 0.0;
    static constexpr int P[3] = {0, 0, 1}, Q[3] = {1, 2, 2};
    for (int sweep = 0; sweep < 64; ++sweep) {
        bool rotated = false;
        for (int r = 0; r < 3; ++r) {
            const int p = P[r], q = Q[r];
            const double apq = a[p][q];
            if (apq == 0.0) continue;
            if (std::abs(apq) <= 1e-18 * (std::abs(a[p][p]) + std::abs(a[q][q]))) {
                a[p][q] = a[q][p] = 0.0;
                continue;
            }
            rotated = true;
            const double theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            const double t = std::abs(theta) > 1e150 ? 0.5 / theta
                                                     : (theta >= 0 ? 1.0 : -1.0) /
                                                           (std::abs(theta) + std::sqrt(theta * theta + 1.0));
            const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
            for (int k = 0; k < 3; ++k) {  // a <- a j
                const double akp = a[k][p], akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for (int k = 0; k < 3; ++k) {  // a <- j^t a
                const double apk = a[p][k], aqk = a[q][k];
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            a[p][q] = a[q][p] = 0.0;
            for (int k = 0; k < 3; ++k) {  // v <- v j
                const double vkp = v[k][p], vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
        if (!rotated) break;
    }
    for (int i = 0; i < 3; ++i) w[i] = a[i][i];
}

// shape of a set of positions (shapes._moment_measurements). returns false if undefined.
bool moment_shape(const double cov_in[3][3], double eigval[3], double eigvec[9], double& r_eff, double& ellip) {
    double a[3][3], w[3], v[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) a[i][j] = 0.5 * (cov_in[i][j] + cov_in[j][i]);
    eigen_sym3(a, w, v);
    int idx[3] = {0, 1, 2};
    std::sort(idx, idx + 3, [&](int x, int y) { return w[x] > w[y]; });
    double lam[3], vec[3][3];
    for (int r = 0; r < 3; ++r) {
        lam[r] = w[idx[r]];
        for (int k = 0; k < 3; ++k) vec[r][k] = v[k][idx[r]];
    }
    const double scale = std::max(std::max(std::abs(lam[0]), std::abs(lam[2])), kTiny);
    const double negative_tolerance = 64.0 * kEps * scale;
    for (double l : lam)
        if (l < -negative_tolerance) return false;
    for (double& l : lam) l = std::max(l, 0.0);
    const double rank_tolerance = 3.0 * kEps * lam[0];
    if (lam[0] <= 0.0 || lam[2] <= rank_tolerance) return false;
    // deterministic sign: the largest-magnitude component is positive
    for (auto& row : vec) {
        int c = 0;
        for (int k = 1; k < 3; ++k)
            if (std::abs(row[k]) > std::abs(row[c])) c = k;
        if (row[c] < 0)
            for (double& x : row) x = -x;
    }
    r_eff = std::sqrt(5.0) * std::exp((std::log(lam[0]) + std::log(lam[1]) + std::log(lam[2])) / 6.0);
    const double ratio = std::clamp((lam[2] + lam[1]) / (lam[1] + lam[0]), 0.0, 1.0);
    ellip = 1.0 - std::pow(ratio, 0.25);
    for (int r = 0; r < 3; ++r) {
        eigval[r] = lam[r];
        for (int k = 0; k < 3; ++k) eigvec[3 * r + k] = vec[r][k];
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------- selection mask
Index SelectionMask::radial_bin(double radius) const {
    // np.searchsorted(edges, radius, side='right') - 1, with the last edge in the last bin
    const auto it = std::upper_bound(radial_edges.begin(), radial_edges.end(), radius);
    Index bin = Index(it - radial_edges.begin()) - 1;
    if (bin == Index(radial_edges.size()) - 1) bin = Index(radial_edges.size()) - 2;
    return bin;
}

Index SelectionMask::n_valid_pixels() const {
    Index count = 0;
    for (std::size_t p = 0; p < pixel_counts.size(); ++p) count += pixel_valid(std::int64_t(p));
    return count;
}

Index SelectionMask::n_valid_bins() const {
    Index count = 0;
    for (Index b = 0; b < n_bins(); ++b) count += bin_valid(b);
    return count;
}

SelectionMask build_selection_mask(const double* ra, const double* dec, const double* xyz, Index n,
                                   std::int64_t nside, int min_randoms_per_pixel, double radial_bin_width,
                                   int min_randoms_per_radial_bin, int threads) {
    if (!healpix::valid_nside(nside)) throw std::invalid_argument("invalid healpix nside");
    if (min_randoms_per_pixel < 1 || min_randoms_per_radial_bin < 1)
        throw std::invalid_argument("minimum random counts must be positive");
    if (!(radial_bin_width > 0) || !std::isfinite(radial_bin_width))
        throw std::invalid_argument("radial_bin_width must be positive");
    if (n <= 0) throw std::invalid_argument("the mask needs at least one random");
    SelectionMask mask;
    mask.nside = nside;
    mask.min_randoms_per_pixel = min_randoms_per_pixel;
    mask.min_randoms_per_radial_bin = min_randoms_per_radial_bin;
    mask.radial_bin_width = radial_bin_width;
    mask.n_rows = n;
    const int nt = resolve_threads(threads);
    const auto npix = std::size_t(healpix::npix(nside));

    // radial index range: floor(radius / width) over valid radii
    Index lo_idx = std::numeric_limits<Index>::max(), hi_idx = std::numeric_limits<Index>::min();
    Index n_ang = 0, n_rad = 0;
#pragma omp parallel for num_threads(nt) reduction(min : lo_idx) reduction(max : hi_idx) \
    reduction(+ : n_ang, n_rad) schedule(static)
    for (Index i = 0; i < n; ++i) {
        if (std::isfinite(ra[i]) && std::isfinite(dec[i]) && dec[i] >= -90.0 && dec[i] <= 90.0) ++n_ang;
        const double radius = norm3(xyz + 3 * i);
        if (std::isfinite(radius) && radius > 0.0) {
            const auto b = Index(std::floor(radius / radial_bin_width));
            lo_idx = std::min(lo_idx, b);
            hi_idx = std::max(hi_idx, b);
            ++n_rad;
        }
    }
    if (n_rad == 0) throw std::invalid_argument("no valid random cartesian radii");
    const Index n_bins = hi_idx - lo_idx + 1;
    if (n_bins > 10'000'000)
        throw std::invalid_argument("the random radius range needs more than 1e7 radial bins");
    mask.n_valid_angular = n_ang;
    mask.n_valid_radial = n_rad;

    mask.pixel_counts.assign(npix, 0);
    mask.radial_counts.assign(std::size_t(n_bins), 0);
#pragma omp parallel num_threads(nt)
    {
        std::vector<std::int64_t> pix(npix, 0), rad(std::size_t(n_bins), 0);
#pragma omp for schedule(static)
        for (Index i = 0; i < n; ++i) {
            if (std::isfinite(ra[i]) && std::isfinite(dec[i]) && dec[i] >= -90.0 && dec[i] <= 90.0)
                ++pix[std::size_t(healpix::radec2pix_ring(nside, ra[i], dec[i]))];
            const double radius = norm3(xyz + 3 * i);
            if (std::isfinite(radius) && radius > 0.0)
                ++rad[std::size_t(Index(std::floor(radius / radial_bin_width)) - lo_idx)];
        }
#pragma omp critical
        {
            for (std::size_t p = 0; p < npix; ++p) mask.pixel_counts[p] += pix[p];
            for (std::size_t b = 0; b < rad.size(); ++b) mask.radial_counts[b] += rad[b];
        }
    }
    mask.radial_edges.resize(std::size_t(n_bins) + 1);
    for (Index b = 0; b <= n_bins; ++b) mask.radial_edges[std::size_t(b)] = double(lo_idx + b) * radial_bin_width;
    mask.invalid_prefix.assign(std::size_t(n_bins) + 1, 0);
    for (Index b = 0; b < n_bins; ++b)
        mask.invalid_prefix[std::size_t(b) + 1] = mask.invalid_prefix[std::size_t(b)] + !mask.bin_valid(b);
    return mask;
}

// ---------------------------------------------------------------- void finder
VoidFinder::VoidFinder(const std::int32_t* ndata, const std::int32_t* nrand, Index n, double r_threshold,
                       int threads)
    : n_(n), r_threshold_(r_threshold) {
    if (n <= 0) throw std::invalid_argument("no points");
    if (!std::isfinite(r_threshold) || r_threshold < -1.0 || r_threshold > 1.0)
        throw std::invalid_argument("r_threshold must lie within [-1, 1]");
    const int nt = resolve_threads(threads);

    // r = (nd - nr) / (nd + nr) only takes the values of the (nd, degree) pairs,
    // so the stable sort by r is a counting sort over the distinct values.
    std::int32_t max_degree = 0;
#pragma omp parallel for num_threads(nt) reduction(max : max_degree) schedule(static)
    for (Index i = 0; i < n; ++i) {
        if (ndata[i] < 0 || nrand[i] < 0) continue;
        max_degree = std::max(max_degree, ndata[i] + nrand[i]);
    }
    const std::int64_t D = max_degree;
    // r value of each (nd, degree) cell, same float64 operations as numpy
    auto cell_of = [D](std::int64_t nd, std::int64_t deg) { return std::size_t(deg * (D + 1) + nd); };
    std::vector<double> cell_r(std::size_t((D + 1) * (D + 1)), 0.0);
    std::vector<double> values;
    for (std::int64_t deg = 1; deg <= D; ++deg)
        for (std::int64_t nd = 0; nd <= deg; ++nd) {
            const double r = double(nd - (deg - nd)) / double(deg);
            cell_r[cell_of(nd, deg)] = r;
            if (r <= r_threshold) values.push_back(r);
        }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    const auto n_buckets = std::size_t(values.size());
    std::vector<std::int32_t> cell_bucket(cell_r.size(), -1);
    for (std::int64_t deg = 1; deg <= D; ++deg)
        for (std::int64_t nd = 0; nd <= deg; ++nd) {
            const double r = cell_r[cell_of(nd, deg)];
            if (r <= r_threshold)
                cell_bucket[cell_of(nd, deg)] =
                    std::int32_t(std::lower_bound(values.begin(), values.end(), r) - values.begin());
        }
    auto bucket_of = [&](Index i) -> std::int32_t {
        const std::int64_t nd = ndata[i], deg = std::int64_t(ndata[i]) + nrand[i];
        if (nd < 0 || nrand[i] < 0 || deg <= 0) return -1;  // r undefined: never selected
        return cell_bucket[cell_of(nd, deg)];
    };

    // stable parallel counting sort: thread t scans the block [lo_t, hi_t) and
    // writes after every earlier thread's points of the same bucket.
    // (cap the per-thread counters at ~2^26 in total)
    const Index max_counter_threads = (Index(1) << 26) / Index(std::max<std::size_t>(1, n_buckets));
    const int nts = int(std::clamp<Index>(max_counter_threads, 1, nt));
    std::vector<Index> counts(n_buckets * std::size_t(nts), 0);
#pragma omp parallel num_threads(nts)
    {
        const int t = omp_get_thread_num();
        Index lo, hi;
        block_range(n, t, nts, lo, hi);
        Index* c = counts.data() + std::size_t(t) * n_buckets;
        for (Index i = lo; i < hi; ++i) {
            const std::int32_t b = bucket_of(i);
            if (b >= 0) ++c[b];
        }
    }
    std::vector<Index> start(n_buckets * std::size_t(nts));
    Index total = 0;
    for (std::size_t b = 0; b < n_buckets; ++b)
        for (int t = 0; t < nts; ++t) {
            start[std::size_t(t) * n_buckets + b] = total;
            total += counts[std::size_t(t) * n_buckets + b];
        }
    if (total > Index(std::numeric_limits<std::int32_t>::max()))
        throw std::runtime_error("more than 2^31 selected points");
    order_.resize(std::size_t(total));
    rank_.assign(std::size_t(n), -1);
#pragma omp parallel num_threads(nts)
    {
        const int t = omp_get_thread_num();
        Index lo, hi;
        block_range(n, t, nts, lo, hi);
        Index* s = start.data() + std::size_t(t) * n_buckets;
        for (Index i = lo; i < hi; ++i) {
            const std::int32_t b = bucket_of(i);
            if (b < 0) continue;
            const Index pos = s[b]++;
            order_[std::size_t(pos)] = i;
            rank_[std::size_t(i)] = std::int32_t(pos);
        }
    }
}

void VoidFinder::merge_duplicates(const std::vector<std::array<Index, 2>>& duplicates) {
    if (!adj_.offsets.empty() || grown_) throw std::logic_error("merge duplicates before set_adjacency()");
    // canonical point of each duplicate, following chains (rare)
    std::vector<std::array<Index, 2>> pairs;
    for (const auto& d : duplicates) {
        if (d[0] < 0 || d[0] >= n_ || d[1] < 0 || d[1] >= n_) throw std::out_of_range("duplicate outside the points");
        if (d[0] != d[1] && rank_[std::size_t(d[0])] >= 0) pairs.push_back(d);
    }
    if (pairs.empty()) return;
    std::sort(pairs.begin(), pairs.end());
    auto canonical = [&](Index c) {
        for (int step = 0; step < 64; ++step) {
            const auto it = std::lower_bound(pairs.begin(), pairs.end(), std::array<Index, 2>{c, -1});
            if (it == pairs.end() || (*it)[0] != c) return c;
            c = (*it)[1];
        }
        throw std::runtime_error("cyclic duplicate chain");
    };
    for (const auto& d : pairs) {
        const Index c = canonical(d[1]);
        if (rank_[std::size_t(c)] < 0) continue;  // same counts: cannot happen
        aliases_.push_back({d[0], c});
    }
    for (const auto& a : aliases_) rank_[std::size_t(a[0])] = -1;
    // compact the processing order and renumber the ranks
    std::size_t w = 0;
    for (std::size_t k = 0; k < order_.size(); ++k) {
        const Index i = order_[k];
        if (rank_[std::size_t(i)] < 0) continue;
        order_[w] = i;
        rank_[std::size_t(i)] = std::int32_t(w);
        ++w;
    }
    order_.resize(w);
}

void VoidFinder::set_adjacency(LowerAdjacency&& adjacency) {
    if (adjacency.offsets.size() != order_.size() + 1)
        throw std::invalid_argument("the adjacency does not match the selected points");
    adj_ = std::move(adjacency);
}

void VoidFinder::release_adjacency() {
    LowerAdjacency().offsets.swap(adj_.offsets);
    std::vector<std::int32_t>().swap(adj_.neighbors);
}

void VoidFinder::grow(int min_members) {
    if (min_members < 1) throw std::invalid_argument("min_members must be positive");
    const Index m = n_selected();
    if (adj_.offsets.size() != std::size_t(m) + 1) throw std::logic_error("set the adjacency before grow()");
    const std::int64_t* off = adj_.offsets.data();
    const std::int32_t* nb = adj_.neighbors.data();
    premask_.assign(std::size_t(m), kNoGroup);
    std::int32_t* grp = premask_.data();
    seed_rank_.clear();
    seed_size_.clear();

    // every lower neighbor was processed before and already has a group, so the
    // group of rank k is the minimum over its row (or a new seed if empty). the
    // loop is a stream over the csr; prefetch the group ids of upcoming rows.
    constexpr Index kAhead = 24;
    for (Index k = 0; k < m; ++k) {
        if (k + kAhead < m)
            for (std::int64_t e = off[k + kAhead], end = off[k + kAhead + 1]; e < end; ++e)
                __builtin_prefetch(grp + nb[e]);
        std::int32_t best = std::numeric_limits<std::int32_t>::max();
        for (std::int64_t e = off[k], end = off[k + 1]; e < end; ++e) best = std::min(best, grp[nb[e]]);
        if (best == std::numeric_limits<std::int32_t>::max()) {
            best = std::int32_t(seed_rank_.size());
            seed_rank_.push_back(k);
            seed_size_.push_back(0);
        }
        grp[k] = best;
        ++seed_size_[std::size_t(best)];
    }

    const Index n_seeds = Index(seed_rank_.size());
    Index retained = 0;
#pragma omp parallel for reduction(+ : retained) schedule(static)
    for (Index g = 0; g < n_seeds; ++g) retained += seed_size_[std::size_t(g)] >= min_members;
    n_groups_premask_ = retained;
#pragma omp parallel for schedule(static)
    for (Index k = 0; k < m; ++k)
        if (seed_size_[std::size_t(grp[k])] < min_members) grp[k] = kNoGroup;
    group_ = premask_;
    border_.assign(std::size_t(n_seeds), 0);
    grown_ = true;
}

Index VoidFinder::n_groups() const {
    std::vector<std::uint8_t> present(seed_rank_.size(), 0);
    for (std::int32_t g : group_)
        if (g >= 0) present[std::size_t(g)] = 1;
    return Index(std::count(present.begin(), present.end(), std::uint8_t(1)));
}

MaskStats VoidFinder::apply_selection_mask(const SelectionMask& mask, const double* xyz, const std::uint8_t* is_data,
                                           const double* ra, const double* dec, double angular_sample_step,
                                           int min_members, int threads) {
    if (!grown_) throw std::logic_error("call grow() before apply_selection_mask()");
    if (adj_.offsets.size() != order_.size() + 1) throw std::logic_error("the adjacency was released");
    if (min_members < 1) throw std::invalid_argument("min_members must be positive");
    if (!(angular_sample_step > 0)) throw std::invalid_argument("angular_sample_step must be positive");
    const int nt = resolve_threads(threads);
    const Index m = n_selected();
    const Index n_seeds = Index(seed_rank_.size());
    const std::int64_t* off = adj_.offsets.data();
    const std::int32_t* nb = adj_.neighbors.data();
    const std::int32_t* pre = premask_.data();
    const double e_lo = mask.radial_edges.front(), e_hi = mask.radial_edges.back();
    MaskStats st;

    // per member: angular usability, valid pixel, valid radial bin
    enum : std::uint8_t { kUsable = 1, kPixelOk = 2, kRadialOk = 4 };
    std::vector<std::uint8_t> pt(std::size_t(m), 0);
    Index n_inv_ang = 0, n_inv_rad = 0;
#pragma omp parallel for num_threads(nt) reduction(+ : n_inv_ang, n_inv_rad) schedule(static)
    for (Index k = 0; k < m; ++k) {
        const std::int32_t g = pre[k];
        if (g < 0) continue;
        const Index i = order_[std::size_t(k)];
        const double* p = xyz + 3 * i;
        const double radius = norm3(p);
        std::uint8_t f = 0;
        // randoms use their (ra, dec) and data their cartesian direction (as watershed.py)
        std::int64_t pixel = -1;
        if (!is_data[i] && ra != nullptr) {
            if (std::isfinite(ra[i]) && std::isfinite(dec[i]) && dec[i] >= -90.0 && dec[i] <= 90.0)
                pixel = healpix::radec2pix_ring(mask.nside, ra[i], dec[i]);
        } else if (finite3(p) && std::isfinite(radius) && radius > 0.0) {
            pixel = healpix::vec2pix_ring(mask.nside, p[0] / radius, p[1] / radius, p[2] / radius);
        }
        if (pixel >= 0) {
            f |= kUsable;
            if (mask.pixel_valid(pixel)) f |= kPixelOk;
        }
        if (finite3(p) && std::isfinite(radius) && radius > 0.0 && radius >= e_lo && radius <= e_hi &&
            mask.bin_valid(mask.radial_bin(radius)))
            f |= kRadialOk;
        pt[std::size_t(k)] = f;
        if (!(f & kPixelOk)) {
            ++n_inv_ang;
            __atomic_fetch_or(&border_[std::size_t(g)], std::uint8_t(kBorderAngularMember), __ATOMIC_RELAXED);
        }
        if (!(f & kRadialOk)) {
            ++n_inv_rad;
            __atomic_fetch_or(&border_[std::size_t(g)], std::uint8_t(kBorderRadialMember), __ATOMIC_RELAXED);
        }
    }
    st.n_invalid_angular_members = n_inv_ang;
    st.n_invalid_radial_members = n_inv_rad;
    auto valid_member = [&](Index k) { return (pt[std::size_t(k)] & (kPixelOk | kRadialOk)) == (kPixelOk | kRadialOk); };

    // great-circle arc between two members sampled every angular_sample_step
    auto angular_crossing = [&](Index ka, Index kb) -> bool {
        if (!((pt[std::size_t(ka)] & kUsable) && (pt[std::size_t(kb)] & kUsable))) return true;
        const double* p0 = xyz + 3 * order_[std::size_t(ka)];
        const double* p1 = xyz + 3 * order_[std::size_t(kb)];
        const double r0 = norm3(p0), r1 = norm3(p1);
        if (!(r0 > kTiny && r1 > kTiny)) return true;
        if (!((pt[std::size_t(ka)] & kPixelOk) && (pt[std::size_t(kb)] & kPixelOk))) return true;
        const double u0[3] = {p0[0] / r0, p0[1] / r0, p0[2] / r0};
        const double u1[3] = {p1[0] / r1, p1[1] / r1, p1[2] / r1};
        const double dot = std::clamp(u0[0] * u1[0] + u0[1] * u1[1] + u0[2] * u1[2], -1.0, 1.0);
        const double angle = std::acos(dot);
        if (!(angle < kPi - 1.0e-10)) return true;  // antipodal: no unique arc
        const auto n_seg = std::max<std::int64_t>(1, std::int64_t(std::ceil(angle / angular_sample_step)));
        const double sin_angle = std::sin(angle);
        for (std::int64_t s = 1; s < n_seg; ++s) {
            const double f = double(s) / double(n_seg);
            double v[3];
            if (std::abs(sin_angle) >= 1.0e-12) {
                const double w0 = std::sin((1.0 - f) * angle) / sin_angle;
                const double w1 = std::sin(f * angle) / sin_angle;
                for (int c = 0; c < 3; ++c) v[c] = w0 * u0[c] + w1 * u1[c];
            } else {
                for (int c = 0; c < 3; ++c) v[c] = (1.0 - f) * u0[c] + f * u1[c];
            }
            const double len = norm3(v);
            if (!(len > kTiny)) return true;
            for (double& c : v) c /= len;
            if (!mask.pixel_valid(healpix::vec2pix_ring(mask.nside, v[0], v[1], v[2]))) return true;
        }
        return false;
    };

    // segment between two members: its radial range must avoid invalid bins
    auto radial_crossing = [&](Index ka, Index kb) -> bool {
        const double* p0 = xyz + 3 * order_[std::size_t(ka)];
        const double* p1 = xyz + 3 * order_[std::size_t(kb)];
        if (!finite3(p0) || !finite3(p1)) return true;
        const double d[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
        const double dsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        double t = 0.0;
        if (dsq > 0.0) t = -(p0[0] * d[0] + p0[1] * d[1] + p0[2] * d[2]) / dsq;
        t = std::clamp(t, 0.0, 1.0);
        const double closest[3] = {p0[0] + t * d[0], p0[1] + t * d[1], p0[2] + t * d[2]};
        const double r_min = norm3(closest), r_max = std::max(norm3(p0), norm3(p1));
        if (!(std::isfinite(r_max) && std::isfinite(r_min) && r_min > 0.0 && r_min >= e_lo && r_max <= e_hi))
            return true;
        const Index first = mask.radial_bin(r_min), last = mask.radial_bin(r_max);
        return mask.invalid_prefix[std::size_t(last) + 1] - mask.invalid_prefix[std::size_t(first)] != 0;
    };

    // internal edges: flag crossings and join the valid ones
    AtomicUnionFind uf(m, nt);
    Index n_internal = 0, n_ang = 0, n_rad = 0;
#pragma omp parallel for num_threads(nt) reduction(+ : n_internal, n_ang, n_rad) schedule(dynamic, 2048)
    for (Index k = 0; k < m; ++k) {
        const std::int32_t g = pre[k];
        if (g < 0) continue;
        for (std::int64_t e = off[k]; e < off[k + 1]; ++e) {
            const Index j = nb[e];
            if (pre[j] != g) continue;
            ++n_internal;
            const bool ang = angular_crossing(k, j);
            const bool rad = radial_crossing(k, j);
            if (ang) {
                ++n_ang;
                __atomic_fetch_or(&border_[std::size_t(g)], std::uint8_t(kBorderAngularEdge), __ATOMIC_RELAXED);
            }
            if (rad) {
                ++n_rad;
                __atomic_fetch_or(&border_[std::size_t(g)], std::uint8_t(kBorderRadialEdge), __ATOMIC_RELAXED);
            }
            if (!ang && !rad && valid_member(k) && valid_member(j)) uf.unite(std::int32_t(k), std::int32_t(j));
        }
    }
    st.n_internal_edges = n_internal;
    st.n_angular_edge_crossings = n_ang;
    st.n_radial_edge_crossings = n_rad;

    // keep the connected component of each group's seed
    std::vector<std::int32_t> seed_root(std::size_t(n_seeds), -1);  // -1: seed invalid
#pragma omp parallel for num_threads(nt) schedule(static)
    for (Index g = 0; g < n_seeds; ++g) {
        const Index s = seed_rank_[std::size_t(g)];
        if (pre[s] >= 0 && valid_member(s)) seed_root[std::size_t(g)] = uf.find(std::int32_t(s));
    }
    std::vector<std::int64_t> component(std::size_t(n_seeds), 0);
    std::vector<std::uint8_t> in_seed(std::size_t(m), 0);
#pragma omp parallel for num_threads(nt) schedule(static)
    for (Index k = 0; k < m; ++k) {
        const std::int32_t g = pre[k];
        if (g < 0 || !valid_member(k) || seed_root[std::size_t(g)] < 0) continue;
        if (uf.find(std::int32_t(k)) == seed_root[std::size_t(g)]) {
            in_seed[std::size_t(k)] = 1;
            __atomic_fetch_add(&component[std::size_t(g)], 1, __ATOMIC_RELAXED);
        }
    }
    std::vector<std::int64_t> premask_size(std::size_t(n_seeds), 0);
    for (Index k = 0; k < m; ++k)
        if (pre[k] >= 0) ++premask_size[std::size_t(pre[k])];

    Index n_affected = 0, n_pruned_groups = 0, n_discarded = 0, n_seed_invalid = 0, n_undersized = 0;
    std::vector<std::uint8_t> accepted(std::size_t(n_seeds), 0);
    for (Index g = 0; g < n_seeds; ++g) {
        if (premask_size[std::size_t(g)] == 0) continue;
        const bool seed_invalid = seed_root[std::size_t(g)] < 0;
        const bool undersized = !seed_invalid && component[std::size_t(g)] < min_members;
        accepted[std::size_t(g)] = !seed_invalid && !undersized;
        const std::int64_t after = accepted[std::size_t(g)] ? component[std::size_t(g)] : 0;
        n_affected += border_[std::size_t(g)] != 0;
        n_seed_invalid += seed_invalid;
        n_undersized += undersized;
        n_pruned_groups += after < premask_size[std::size_t(g)];
        n_discarded += after == 0;
    }
    st.n_groups_affected = n_affected;
    st.n_groups_pruned = n_pruned_groups;
    st.n_groups_discarded = n_discarded;
    st.n_groups_seed_invalid = n_seed_invalid;
    st.n_groups_undersized = n_undersized;

    Index n_pruned = 0, n_inv = 0, n_disc = 0, n_small = 0;
#pragma omp parallel for num_threads(nt) reduction(+ : n_pruned, n_inv, n_disc, n_small) schedule(static)
    for (Index k = 0; k < m; ++k) {
        const std::int32_t g = pre[k];
        if (g < 0) continue;
        const bool keep = in_seed[std::size_t(k)] && accepted[std::size_t(g)];
        group_[std::size_t(k)] = keep ? g : kNoGroup;
        if (keep) continue;
        ++n_pruned;
        const bool valid = valid_member(k);
        const bool seed_invalid = seed_root[std::size_t(g)] < 0;
        n_inv += !valid;
        n_disc += valid && !in_seed[std::size_t(k)] && !seed_invalid;
        n_small += in_seed[std::size_t(k)] && !accepted[std::size_t(g)];
    }
    st.n_pruned_members = n_pruned;
    st.n_invalid_members_pruned = n_inv;
    st.n_disconnected_members = n_disc;
    st.n_small_component_members = n_small;
    return st;
}

Index VoidFinder::flag_box_border(const double* xyz, const Box& box, double buffer, int threads) {
    if (!grown_) throw std::logic_error("call grow() before flag_box_border()");
    if (!(buffer >= 0)) throw std::invalid_argument("buffer must be non-negative");
    if (buffer == 0) return 0;
    const Index m = n_selected();
#pragma omp parallel for num_threads(resolve_threads(threads)) schedule(static)
    for (Index k = 0; k < m; ++k) {
        const std::int32_t g = premask_[std::size_t(k)];
        if (g < 0) continue;
        const double* p = xyz + 3 * order_[std::size_t(k)];
        bool near = false;
        for (int c = 0; c < 3; ++c) near |= (p[c] - box.lo[c] <= buffer) || (box.hi[c] - p[c] <= buffer);
        if (near) __atomic_fetch_or(&border_[std::size_t(g)], std::uint8_t(kBorderBoxFace), __ATOMIC_RELAXED);
    }
    Index count = 0;
    for (std::uint8_t b : border_) count += (b & kBorderBoxFace) != 0;
    return count;
}

VoidShapes VoidFinder::shapes(const double* xyz, const std::uint8_t* is_data, double scale, const Box* periodic,
                              int threads) const {
    if (!grown_) throw std::logic_error("call grow() before shapes()");
    if (!(scale > 0) || !std::isfinite(scale)) throw std::invalid_argument("scale must be positive");
    const int nt = resolve_threads(threads);
    const Index m = n_selected();
    const Index n_seeds = Index(seed_rank_.size());

    // members per group (data and randoms), then the random members of each
    // group contiguous and in rank order
    std::vector<std::int64_t> n_data(std::size_t(n_seeds), 0), n_rand(std::size_t(n_seeds), 0);
    for (Index k = 0; k < m; ++k) {
        const std::int32_t g = group_[std::size_t(k)];
        if (g >= 0) ++(is_data[order_[std::size_t(k)]] ? n_data : n_rand)[std::size_t(g)];
    }
    auto alias_group = [&](const std::array<Index, 2>& a) { return group_[std::size_t(rank_[std::size_t(a[1])])]; };
    for (const auto& a : aliases_) {
        const std::int32_t g = alias_group(a);
        if (g >= 0) ++(is_data[a[0]] ? n_data : n_rand)[std::size_t(g)];
    }
    VoidShapes out;
    std::vector<std::int64_t> row_of(std::size_t(n_seeds), -1);
    for (Index g = 0; g < n_seeds; ++g)
        if (n_data[std::size_t(g)] + n_rand[std::size_t(g)] > 0) {
            row_of[std::size_t(g)] = std::int64_t(out.group_id.size());
            out.group_id.push_back(std::int32_t(g));
        }
    const std::size_t rows = out.group_id.size();
    std::vector<std::int64_t> start(rows + 1, 0);
    for (std::size_t r = 0; r < rows; ++r) start[r + 1] = start[r] + n_rand[std::size_t(out.group_id[r])];
    std::vector<Index> members(static_cast<std::size_t>(start[rows]));
    {
        std::vector<std::int64_t> fill(start.begin(), start.end() - 1);
        for (Index k = 0; k < m; ++k) {
            const std::int32_t g = group_[std::size_t(k)];
            const Index i = order_[std::size_t(k)];
            if (g >= 0 && !is_data[i]) members[std::size_t(fill[std::size_t(row_of[std::size_t(g)])]++)] = i;
        }
        for (const auto& a : aliases_) {
            const std::int32_t g = alias_group(a);
            if (g >= 0 && !is_data[a[0]]) members[std::size_t(fill[std::size_t(row_of[std::size_t(g)])]++)] = a[0];
        }
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    out.n_members.resize(rows);
    out.n_data.resize(rows);
    out.n_random.resize(rows);
    out.center.assign(3 * rows, nan);
    out.eigval.assign(3 * rows, nan);
    out.eigvec.assign(9 * rows, nan);
    out.r_eff.assign(rows, nan);
    out.ellip.assign(rows, nan);
    out.wrap_ambiguous.assign(rows, 0);
    double L[3] = {0, 0, 0};
    if (periodic)
        for (int c = 0; c < 3; ++c) L[c] = periodic->hi[c] - periodic->lo[c];

#pragma omp parallel for num_threads(nt) schedule(dynamic, 64)
    for (std::size_t r = 0; r < rows; ++r) {
        const auto g = std::size_t(out.group_id[r]);
        out.n_data[r] = n_data[g];
        out.n_random[r] = n_rand[g];
        out.n_members[r] = n_data[g] + n_rand[g];
        const std::int64_t count = start[r + 1] - start[r];
        if (count == 0) continue;
        const Index* mem = members.data() + start[r];
        // positions unwrapped around the first member in periodic boxes
        const double* anchor = xyz + 3 * mem[0];
        auto position = [&](Index i, double q[3]) {
            const double* p = xyz + 3 * i;
            for (int c = 0; c < 3; ++c) {
                double x = p[c];
                if (periodic) {
                    const double d = x - anchor[c];
                    x = anchor[c] + d - L[c] * std::nearbyint(d / L[c]);
                }
                q[c] = x * scale;
            }
        };
        double mean[3] = {0, 0, 0}, lo[3], hi[3];
        for (int c = 0; c < 3; ++c) {
            lo[c] = std::numeric_limits<double>::infinity();
            hi[c] = -lo[c];
        }
        for (std::int64_t a = 0; a < count; ++a) {
            double q[3];
            position(mem[a], q);
            for (int c = 0; c < 3; ++c) {
                mean[c] += q[c];
                lo[c] = std::min(lo[c], q[c]);
                hi[c] = std::max(hi[c], q[c]);
            }
        }
        if (periodic)
            for (int c = 0; c < 3; ++c)
                if (hi[c] - lo[c] > 0.5 * L[c] * scale) out.wrap_ambiguous[r] = 1;
        for (double& c : mean) c /= double(count);
        double cov[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        for (std::int64_t a = 0; a < count; ++a) {
            double q[3];
            position(mem[a], q);
            const double d[3] = {q[0] - mean[0], q[1] - mean[1], q[2] - mean[2]};
            for (int i = 0; i < 3; ++i)
                for (int j = i; j < 3; ++j) cov[i][j] += d[i] * d[j];
        }
        for (int i = 0; i < 3; ++i)
            for (int j = i; j < 3; ++j) cov[j][i] = cov[i][j] = cov[i][j] / double(count);
        double center[3] = {mean[0], mean[1], mean[2]};
        if (periodic)
            for (int c = 0; c < 3; ++c) {
                const double lo = periodic->lo[c] * scale, len = L[c] * scale;
                center[c] = lo + (center[c] - lo) - len * std::floor((center[c] - lo) / len);
                if (center[c] >= lo + len) center[c] = lo;
            }
        for (int c = 0; c < 3; ++c) out.center[3 * r + std::size_t(c)] = center[c];
        moment_shape(cov, &out.eigval[3 * r], &out.eigvec[9 * r], out.r_eff[r], out.ellip[r]);
    }
    return out;
}

void VoidFinder::membership(std::int32_t* premask, std::int32_t* group, std::uint8_t* flags, int threads) const {
    if (!grown_) throw std::logic_error("call grow() before membership()");
    const int nt = resolve_threads(threads);
#pragma omp parallel for num_threads(nt) schedule(static)
    for (Index i = 0; i < n_; ++i) {
        premask[i] = kNoGroup;
        group[i] = kNoGroup;
        flags[i] = 0;
    }
    const Index m = n_selected();
#pragma omp parallel for num_threads(nt) schedule(static)
    for (Index k = 0; k < m; ++k) {
        const Index i = order_[std::size_t(k)];
        const std::int32_t pre = premask_[std::size_t(k)], fin = group_[std::size_t(k)];
        premask[i] = pre;
        group[i] = fin;
        std::uint8_t f = kMemberSelected;
        if (pre >= 0 && fin < 0) f |= kMemberPruned;
        if (pre >= 0 && border_[std::size_t(pre)]) f |= kMemberBorder;
        flags[i] = f;
    }
    for (const auto& a : aliases_) {  // merged duplicates follow their canonical point
        premask[a[0]] = premask[a[1]];
        group[a[0]] = group[a[1]];
        flags[a[0]] = flags[a[1]];
    }
}

}  // namespace astra

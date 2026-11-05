// consensus catalog of voids pooled from several realizations (consensus.py).
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "astra/voids.hpp"

namespace astra {

double contained_volume_fraction(double distance, double container, double sphere) {
    constexpr double kPi = 3.14159265358979323846;
    double fraction = 0.0;
    if (distance <= std::abs(container - sphere)) {
        fraction = sphere <= container ? 1.0 : std::pow(container / sphere, 3);
    } else if (distance < container + sphere) {
        const double d = distance, big = container, small = sphere;
        const double intersection = kPi * (big + small - d) * (big + small - d) *
                                    (d * d + 2.0 * d * small - 3.0 * small * small + 2.0 * d * big +
                                     6.0 * small * big - 3.0 * big * big) /
                                    (12.0 * d);
        fraction = intersection / ((4.0 / 3.0) * kPi * small * small * small);
    }
    return std::clamp(fraction, 0.0, 1.0);
}

namespace {

// uniform grid of sphere centers for fixed-radius queries, optionally periodic.
class CenterGrid {
public:
    CenterGrid(const double* centers, const std::vector<Index>& items, double cell, const ConsensusOptions& opt)
        : periodic_(opt.periodic) {
        const std::size_t n = items.size();
        double lo[3], hi[3];
        if (periodic_) {
            for (int c = 0; c < 3; ++c) {
                lo[c] = opt.box.lo[c];
                hi[c] = opt.box.hi[c];
                L_[c] = hi[c] - lo[c];
                if (!(L_[c] > 0)) throw std::invalid_argument("invalid periodic box");
            }
        } else {
            for (int c = 0; c < 3; ++c) {
                lo[c] = std::numeric_limits<double>::infinity();
                hi[c] = -lo[c];
            }
            for (Index i : items)
                for (int c = 0; c < 3; ++c) {
                    lo[c] = std::min(lo[c], centers[3 * i + c]);
                    hi[c] = std::max(hi[c], centers[3 * i + c]);
                }
            if (n == 0)
                for (int c = 0; c < 3; ++c) lo[c] = hi[c] = 0;
        }
        // at most 2^24 cells
        cell = std::max(cell, 1e-12);
        for (;;) {
            double total = 1;
            for (int c = 0; c < 3; ++c) {
                const double extent = hi[c] - lo[c];
                dims_[c] = periodic_ ? std::max<Index>(1, Index(std::floor(extent / cell)))
                                     : std::max<Index>(1, Index(std::ceil(extent / cell)));
                total *= double(dims_[c]);
            }
            if (total <= double(1 << 24)) break;
            cell *= 1.25;
        }
        for (int c = 0; c < 3; ++c) {
            lo_[c] = lo[c];
            size_[c] = periodic_ ? L_[c] / double(dims_[c]) : std::max(cell, 1e-300);
        }
        const std::size_t ncells = std::size_t(dims_[0] * dims_[1] * dims_[2]);
        start_.assign(ncells + 1, 0);
        std::vector<std::size_t> cell_of(n);
        for (std::size_t a = 0; a < n; ++a) {
            const double* p = centers + 3 * items[a];
            Index ix[3];
            for (int c = 0; c < 3; ++c) ix[c] = coord_cell(p[c], c);
            cell_of[a] = flat(ix[0], ix[1], ix[2]);
            ++start_[cell_of[a] + 1];
        }
        for (std::size_t c = 0; c < ncells; ++c) start_[c + 1] += start_[c];
        ids_.resize(n);
        xyz_.resize(3 * n);
        std::vector<std::size_t> fill(start_.begin(), start_.end() - 1);
        for (std::size_t a = 0; a < n; ++a) {
            const std::size_t slot = fill[cell_of[a]]++;
            ids_[slot] = items[a];
            for (int c = 0; c < 3; ++c) xyz_[3 * slot + c] = centers[3 * items[a] + c];
        }
    }

    // call f(id, distance) for every center within `radius` of p (distance <= radius).
    template <class F>
    void query(const double* p, double radius, F&& f) const {
        Index a[3], b[3];
        for (int c = 0; c < 3; ++c) {
            a[c] = Index(std::floor((p[c] - radius - lo_[c]) / size_[c]));
            b[c] = Index(std::floor((p[c] + radius - lo_[c]) / size_[c]));
            if (periodic_) {
                if (b[c] - a[c] + 1 >= dims_[c]) {
                    a[c] = 0;
                    b[c] = dims_[c] - 1;
                }
            } else {
                a[c] = std::max<Index>(a[c], 0);
                b[c] = std::min<Index>(b[c], dims_[c] - 1);
                if (a[c] > b[c]) return;
            }
        }
        for (Index x = a[0]; x <= b[0]; ++x)
            for (Index y = a[1]; y <= b[1]; ++y)
                for (Index z = a[2]; z <= b[2]; ++z) {
                    const std::size_t cell = flat(wrap(x, 0), wrap(y, 1), wrap(z, 2));
                    for (std::size_t s = start_[cell]; s < start_[cell + 1]; ++s) {
                        double d2 = 0;
                        for (int c = 0; c < 3; ++c) {
                            double d = xyz_[3 * s + c] - p[c];
                            if (periodic_) d -= L_[c] * std::nearbyint(d / L_[c]);
                            d2 += d * d;
                        }
                        const double d = std::sqrt(d2);
                        if (d <= radius) f(ids_[s], d);
                    }
                }
    }

private:
    Index coord_cell(double x, int c) const {
        if (periodic_) x = lo_[c] + (x - lo_[c]) - L_[c] * std::floor((x - lo_[c]) / L_[c]);
        const Index i = Index(std::floor((x - lo_[c]) / size_[c]));
        return std::clamp<Index>(i, 0, dims_[c] - 1);
    }
    Index wrap(Index i, int c) const {
        if (!periodic_) return i;
        i %= dims_[c];
        return i < 0 ? i + dims_[c] : i;
    }
    std::size_t flat(Index x, Index y, Index z) const { return std::size_t((x * dims_[1] + y) * dims_[2] + z); }

    bool periodic_;
    double lo_[3] = {0, 0, 0}, size_[3] = {1, 1, 1}, L_[3] = {0, 0, 0};
    Index dims_[3] = {1, 1, 1};
    std::vector<std::size_t> start_;
    std::vector<Index> ids_;
    std::vector<double> xyz_;
};

// indices sorted by decreasing radius, ties by increasing index (np.argsort(-r, kind='stable')).
std::vector<Index> by_decreasing_radius(const std::vector<double>& radii) {
    std::vector<Index> order(radii.size());
    std::iota(order.begin(), order.end(), Index(0));
    std::stable_sort(order.begin(), order.end(), [&](Index a, Index b) { return radii[std::size_t(a)] > radii[std::size_t(b)]; });
    return order;
}

double median_radius(std::vector<double> radii) {
    if (radii.empty()) return 1.0;
    auto mid = radii.begin() + std::ptrdiff_t(radii.size() / 2);
    std::nth_element(radii.begin(), mid, radii.end());
    return *mid;
}

// group of a seed: its members plus the representative chosen from them.
struct SeedGroup {
    std::vector<Index> members;
    Index representative = -1;
    std::int32_t support = 0;
};

// greedy largest-first sweep: evaluate the groups of a batch of seeds in parallel
// (a group does not depend on what was absorbed), then consume them in order:
// a seed absorbed by an earlier group is skipped, otherwise its group is
// accepted and all its members are absorbed.
template <class Evaluate, class Accept>
void greedy_sweep(const std::vector<Index>& order, Index batch, int threads, Evaluate&& evaluate, Accept&& accept) {
    std::vector<std::uint8_t> absorbed(order.size(), 0);
    std::vector<Index> seeds;
    std::vector<SeedGroup> groups;
    for (std::size_t start = 0; start < order.size(); start += std::size_t(batch)) {
        const std::size_t end = std::min(order.size(), start + std::size_t(batch));
        seeds.clear();
        for (std::size_t a = start; a < end; ++a)
            if (!absorbed[std::size_t(order[a])]) seeds.push_back(order[a]);
        groups.resize(seeds.size());
#pragma omp parallel for num_threads(threads) schedule(dynamic, 16)
        for (std::size_t s = 0; s < seeds.size(); ++s) {
            groups[s].members.clear();
            evaluate(seeds[s], groups[s]);
        }
        for (std::size_t s = 0; s < seeds.size(); ++s) {
            if (absorbed[std::size_t(seeds[s])]) continue;
            accept(seeds[s], groups[s]);
            for (Index g : groups[s].members) absorbed[std::size_t(g)] = 1;
        }
    }
}

// spheres whose volume fraction inside the seed sphere exceeds vol_frac (and the
// seed itself). for vol_frac >= 0.5 such a sphere has its center inside the seed.
void qualifying(const CenterGrid& grid, const double* centers, const std::vector<double>& radii, Index seed,
                double vol_frac, std::vector<Index>& group) {
    const double R = radii[std::size_t(seed)];
    grid.query(centers + 3 * seed, R, [&](Index j, double d) {
        if (j == seed || contained_volume_fraction(d, R, radii[std::size_t(j)]) > vol_frac) group.push_back(j);
    });
    if (std::find(group.begin(), group.end(), seed) == group.end()) group.push_back(seed);
}

}  // namespace

ConsensusResult void_consensus(const double* centers, const double* radii, const std::int32_t* origins, Index n,
                               int n_runs, const ConsensusOptions& opt, int threads) {
    if (n < 1) throw std::invalid_argument("the pooled catalog is empty");
    if (n_runs < 1) throw std::invalid_argument("n_runs must be positive");
    if (!(opt.vol_frac >= 0.5 && opt.vol_frac <= 1.0)) throw std::invalid_argument("vol_frac must lie within [0.5, 1]");
    if (opt.batch < 1) throw std::invalid_argument("batch must be positive");
    for (Index i = 0; i < n; ++i) {
        if (!(radii[i] > 0) || !std::isfinite(radii[i])) throw std::invalid_argument("radii must be finite and positive");
        if (origins[i] < 0 || origins[i] >= n_runs) throw std::invalid_argument("origins must lie in [0, n_runs)");
        for (int c = 0; c < 3; ++c)
            if (!std::isfinite(centers[3 * i + c])) throw std::invalid_argument("centers must be finite");
    }
    const int nt = resolve_threads(threads);
    const double vol_frac = opt.vol_frac;

    // steps 2-4: groups of pooled spheres, one representative per group: each
    // realization votes with its largest member (ties: lowest index) and the
    // representative is the lower-median-radius vote.
    const std::vector<double> r(radii, radii + n);
    std::vector<Index> all(static_cast<std::size_t>(n));
    std::iota(all.begin(), all.end(), Index(0));
    ConsensusResult out;
    std::vector<Index> medoids;
    std::vector<std::int32_t> support;
    {
        const CenterGrid grid(centers, all, median_radius(r), opt);
        greedy_sweep(
            by_decreasing_radius(r), opt.batch, nt,
            [&](Index seed, SeedGroup& group) {
                qualifying(grid, centers, r, seed, vol_frac, group.members);
                std::vector<Index> best(std::size_t(n_runs), -1);
                for (Index j : group.members) {
                    Index& b = best[std::size_t(origins[j])];
                    if (b < 0 || r[std::size_t(j)] > r[std::size_t(b)] || (r[std::size_t(j)] == r[std::size_t(b)] && j < b))
                        b = j;
                }
                std::vector<Index> votes;
                for (Index b : best)
                    if (b >= 0) votes.push_back(b);
                std::sort(votes.begin(), votes.end(), [&](Index a, Index b) {
                    return r[std::size_t(a)] < r[std::size_t(b)] || (r[std::size_t(a)] == r[std::size_t(b)] && a < b);
                });
                group.representative = votes[(votes.size() - 1) / 2];
                group.support = std::int32_t(votes.size());
            },
            [&](Index, const SeedGroup& group) {
                medoids.push_back(group.representative);
                support.push_back(group.support);
            });
    }
    out.n_groups = Index(medoids.size());

    // step 5: prune overlapping representatives, keeping the largest of each
    // group (ties: lowest representative index). membership is inclusive, so an
    // already kept representative can be picked again: it is kept only once.
    std::vector<double> rep_r(medoids.size()), rep_c(3 * medoids.size());
    for (std::size_t a = 0; a < medoids.size(); ++a) {
        rep_r[a] = r[std::size_t(medoids[a])];
        for (int c = 0; c < 3; ++c) rep_c[3 * a + std::size_t(c)] = centers[3 * medoids[a] + c];
    }
    std::vector<Index> reps(medoids.size());
    std::iota(reps.begin(), reps.end(), Index(0));
    const CenterGrid grid(rep_c.data(), reps, median_radius(rep_r), opt);
    std::vector<std::uint8_t> kept(medoids.size(), 0);
    greedy_sweep(
        by_decreasing_radius(rep_r), opt.batch, nt,
        [&](Index seed, SeedGroup& group) {
            qualifying(grid, rep_c.data(), rep_r, seed, vol_frac, group.members);
            Index keep = group.members.front();
            for (Index j : group.members)
                if (rep_r[std::size_t(j)] > rep_r[std::size_t(keep)] ||
                    (rep_r[std::size_t(j)] == rep_r[std::size_t(keep)] && j < keep))
                    keep = j;
            group.representative = keep;
        },
        [&](Index, const SeedGroup& group) {
            const Index keep = group.representative;
            if (kept[std::size_t(keep)]) {
                ++out.n_duplicates;
                return;
            }
            kept[std::size_t(keep)] = 1;
            out.rows.push_back(medoids[std::size_t(keep)]);
            out.support.push_back(support[std::size_t(keep)]);
            out.n_absorbed.push_back(std::int32_t(group.members.size()) - 1);
        });
    return out;
}

}  // namespace astra

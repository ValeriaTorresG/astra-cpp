#include "astra/delaunay.hpp"

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Spatial_sort_traits_adapter_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/spatial_sort.h>
#include <boost/property_map/property_map.hpp>

#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#endif

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace astra {
namespace {

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = K::Point_3;

// point index in the combined input (originals [0, n), ghosts
// [n, n + n_ghosts)); -1 until the vertex has an owner, allowing
// a new vertex to be distinguished from an existing one (duplicate point).
struct VertexInfo {
    Index id = -1;
};

#ifdef CGAL_LINKED_WITH_TBB
using Concurrency = CGAL::Parallel_tag;
#else
using Concurrency = CGAL::Sequential_tag;
#endif
using Vb = CGAL::Triangulation_vertex_base_with_info_3<VertexInfo, K>;
using Cb = CGAL::Delaunay_triangulation_cell_base_3<K>;
using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb, Concurrency>;
using DT = CGAL::Delaunay_triangulation_3<K, Tds>;
using Vertex_handle = DT::Vertex_handle;
using Cell_handle = DT::Cell_handle;
using Duplicates = std::vector<std::array<Index, 2>>;

// periodic copy of an original point shifted by one box length.
struct Ghost {
    double x, y, z;
    Index orig;
};

// access points without copying: read originals from the input array
// and ghosts from their own vector.
struct PointSource {
    const double* xyz;
    std::size_t n;
    const Ghost* ghosts;

    Point operator()(std::size_t i) const {
        if (i < n) return Point(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
        const Ghost& g = ghosts[i - n];
        return Point(g.x, g.y, g.z);
    }
};

// index -> point property map for cgal spatial sorting.
template <class Key>
struct SourcePmap {
    using key_type = Key;
    using value_type = Point;
    using reference = Point;
    using category = boost::readable_property_map_tag;
    const PointSource* src;
    friend Point get(const SourcePmap& m, Key k) { return (*m.src)(std::size_t(k)); }
};

// assign the index to the vertex returned by insert(); if it already
// had an owner, the point is an exact duplicate.
inline void claim(Vertex_handle v, Index i, Duplicates& dups) {
    if (v == Vertex_handle()) return;
    if (v->info().id < 0) v->info().id = i;
    else if (v->info().id != i) dups.push_back({i, v->info().id});
}

// copies of points within `m` of the [lo, hi) faces, shifted by ±l on
// each axis (up to 7 per corner point). order is deterministic for a
// given thread count.
std::vector<Ghost> make_ghosts(const double* xyz, Index n, const std::array<double, 3>& lo,
                               const std::array<double, 3>& hi, double m, int threads) {
    const int nt = resolve_threads(threads);
    std::vector<std::vector<Ghost>> parts(nt);
#pragma omp parallel num_threads(nt)
    {
        auto& local = parts[omp_get_thread_num()];
#pragma omp for schedule(static)
        for (Index i = 0; i < n; ++i) {
            const double* p = xyz + 3 * i;
            // possible shifts per axis: always 0; +l near lo; -l near hi.
            int opts[3][3], nopt[3];
            for (int k = 0; k < 3; ++k) {
                nopt[k] = 0;
                opts[k][nopt[k]++] = 0;
                if (p[k] < lo[k] + m) opts[k][nopt[k]++] = 1;
                if (p[k] >= hi[k] - m) opts[k][nopt[k]++] = -1;
            }
            if (nopt[0] == 1 && nopt[1] == 1 && nopt[2] == 1) continue;
            for (int a = 0; a < nopt[0]; ++a)
                for (int b = 0; b < nopt[1]; ++b)
                    for (int c = 0; c < nopt[2]; ++c) {
                        const int s[3] = {opts[0][a], opts[1][b], opts[2][c]};
                        if (s[0] == 0 && s[1] == 0 && s[2] == 0) continue;
                        local.push_back({p[0] + s[0] * (hi[0] - lo[0]), p[1] + s[1] * (hi[1] - lo[1]),
                                         p[2] + s[2] * (hi[2] - lo[2]), i});
                    }
        }
    }
    std::size_t total = 0;
    for (const auto& part : parts) total += part.size();
    std::vector<Ghost> out;
    out.reserve(total);
    for (auto& part : parts) {
        out.insert(out.end(), part.begin(), part.end());
        std::vector<Ghost>().swap(part);
    }
    return out;
}

// open-addressed key set cleared in o(1) by changing generation.
// one instance per thread, reused at each vertex.
class KeySet {
public:
    KeySet() { reset_capacity(256); }

    void clear() {
        size_ = 0;
        if (++gen_ == 0) {
            std::fill(stamp_.begin(), stamp_.end(), 0u);
            gen_ = 1;
        }
    }

    // true if key was absent from the set.
    bool insert(std::uintptr_t key) {
        if (2 * (size_ + 1) > keys_.size()) grow();
        std::size_t h = slot(key);
        while (stamp_[h] == gen_) {
            if (keys_[h] == key) return false;
            h = (h + 1) & mask_;
        }
        stamp_[h] = gen_;
        keys_[h] = key;
        ++size_;
        return true;
    }

private:
    std::size_t slot(std::uintptr_t key) const {
        return std::size_t((key * 0x9E3779B97F4A7C15ULL) >> shift_);
    }
    void reset_capacity(std::size_t cap) {
        keys_.assign(cap, 0);
        stamp_.assign(cap, 0u);
        mask_ = cap - 1;
        shift_ = 64 - __builtin_ctzll(cap);
        gen_ = 1;
        size_ = 0;
    }
    void grow() {
        std::vector<std::uintptr_t> live;
        for (std::size_t i = 0; i < keys_.size(); ++i)
            if (stamp_[i] == gen_) live.push_back(keys_[i]);
        reset_capacity(2 * keys_.size());
        for (std::uintptr_t key : live) insert(key);
    }

    std::vector<std::uintptr_t> keys_;
    std::vector<std::uint32_t> stamp_;
    std::size_t mask_ = 0, size_ = 0;
    int shift_ = 0;
    std::uint32_t gen_ = 1;
};

struct NeighborScratch {
    std::vector<Cell_handle> cells;  // bfs queue
    KeySet seen_cells, seen_points;
};

}  // namespace

int auto_lock_grid_cells(Index n) {
    // calibrated on a cut-sky mock: 256 cells per axis is optimal for 6e6 points.
    // the 1024 cap limits the grid to 1024^3 4-byte integers (4 gib).
    const double cells = 1.4 * std::cbrt(double(n));
    return int(std::clamp(cells, 64.0, 1024.0));
}

struct Delaunay3::Impl {
    Index n = 0;
    bool parallel = false;
    int grid_cells = 0;
    DelaunayOptions opt;
    std::vector<Index> ghost_orig;  // original point for each ghost
#ifdef CGAL_LINKED_WITH_TBB
    std::unique_ptr<DT::Lock_data_structure> locks;
#endif
    DT dt;
    Duplicates duplicates;

    Impl(const double* xyz, Index n_points, const DelaunayOptions& options) : n(n_points), opt(options) {
        if (n < 4) throw std::invalid_argument("at least 4 points are required");
        std::vector<Ghost> ghosts;
        if (opt.periodic) {
            for (int k = 0; k < 3; ++k)
                if (!(opt.box_hi[k] > opt.box_lo[k]) || !(opt.ghost_margin > 0) ||
                    opt.ghost_margin >= opt.box_hi[k] - opt.box_lo[k])
                    throw std::invalid_argument("invalid periodic box or ghost margin");
            ghosts = make_ghosts(xyz, n, opt.box_lo, opt.box_hi, opt.ghost_margin, opt.threads);
            ghost_orig.resize(ghosts.size());
            for (std::size_t g = 0; g < ghosts.size(); ++g) ghost_orig[g] = ghosts[g].orig;
        }
        const PointSource src{xyz, std::size_t(n), ghosts.data()};
        const std::size_t total = std::size_t(n) + ghosts.size();
        // use 32-bit indices when possible: save 4 bytes per point during sorting.
        if (total <= std::numeric_limits<std::uint32_t>::max()) build<std::uint32_t>(src, total);
        else build<std::size_t>(src, total);
    }

    Index orig(Index id) const { return id < n ? id : ghost_orig[std::size_t(id - n)]; }

    template <class Key>
    void build(const PointSource& src, std::size_t total) {
        std::vector<Key> order(total);
        std::iota(order.begin(), order.end(), Key(0));
        using Traits = CGAL::Spatial_sort_traits_adapter_3<K, SourcePmap<Key>>;
        const Traits traits(SourcePmap<Key>{&src});

        const int threads = resolve_threads(opt.threads);
#ifdef CGAL_LINKED_WITH_TBB
        parallel = threads > 1;
        if (parallel) {
            grid_cells = opt.lock_grid_cells > 0 ? opt.lock_grid_cells : auto_lock_grid_cells(Index(total));
            tbb::task_arena arena(threads);
            arena.execute([&] {
                CGAL::spatial_sort<CGAL::Parallel_tag>(order.begin(), order.end(), traits);
                insert_parallel(src, order, threads);
            });
            return;
        }
#endif
        CGAL::spatial_sort<CGAL::Sequential_tag>(order.begin(), order.end(), traits);
        Vertex_handle hint;
        for (Key i : order) {
            hint = dt.insert(src(i), hint);
            claim(hint, Index(i), duplicates);
        }
    }

#ifdef CGAL_LINKED_WITH_TBB
    CGAL::Bbox_3 bounding_box(const PointSource& src, std::size_t total, int threads) const {
        double lo[3] = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity()};
        double hi[3] = {-lo[0], -lo[1], -lo[2]};
#pragma omp parallel for num_threads(threads) reduction(min : lo[:3]) reduction(max : hi[:3])
        for (std::size_t i = 0; i < total; ++i) {
            const Point p = src(i);
            for (int k = 0; k < 3; ++k) {
                lo[k] = std::min(lo[k], p[k]);
                hi[k] = std::max(hi[k], p[k]);
            }
        }
        const double pad = 1e-6 * std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]}) + 1e-9;
        return CGAL::Bbox_3(lo[0] - pad, lo[1] - pad, lo[2] - pad, hi[0] + pad, hi[1] + pad, hi[2] + pad);
    }

    // like cgal parallel insertion (delaunay_triangulation_3::insert_with_info),
    // but assign the index before releasing locks and record duplicates.
    template <class Key>
    void insert_parallel(const PointSource& src, const std::vector<Key>& order, int threads) {
        locks = std::make_unique<DT::Lock_data_structure>(bounding_box(src, order.size(), threads), grid_cells);
        dt.set_lock_data_structure(locks.get());

        // insert initial points sequentially until the triangulation is 3d.
        std::size_t start = 0;
        Vertex_handle hint;
        while (start < order.size() && (start < 100 || dt.dimension() < 3)) {
            hint = dt.insert(src(order[start]), hint);
            claim(hint, Index(order[start]), duplicates);
            ++start;
        }
        dt.unlock_all_elements();

        tbb::enumerable_thread_specific<Vertex_handle> tls_hint(hint);
        tbb::enumerable_thread_specific<Duplicates> tls_dups;
        tbb::parallel_for(tbb::blocked_range<std::size_t>(start, order.size()),
                          [&](const tbb::blocked_range<std::size_t>& range) {
            Vertex_handle& h = tls_hint.local();
            Duplicates& dups = tls_dups.local();
            for (std::size_t k = range.begin(); k != range.end(); ++k) {
                const std::size_t i = order[k];
                const Point p = src(i);
                for (bool done = false; !done;) {
                    if (dt.try_lock_vertex(h) && dt.try_lock_point(p)) {
                        bool could_lock_zone = false;
                        const Vertex_handle v = dt.insert(p, h, &could_lock_zone);
                        if (could_lock_zone) {
                            claim(v, Index(i), dups);
                            h = v;
                            done = true;
                        }
                    }
                    dt.unlock_all_elements();
                }
            }
        });
        for (const auto& d : tls_dups) duplicates.insert(duplicates.end(), d.begin(), d.end());
        dt.set_lock_data_structure(nullptr);
        locks.reset();
    }
#endif

    // true if the circumsphere of c fits inside the box expanded by the margin,
    // meaning c is a tetrahedron in the exact periodic triangulation.
    bool circumsphere_inside(Cell_handle c) const {
        const Point center = CGAL::circumcenter(c->vertex(0)->point(), c->vertex(1)->point(),
                                                c->vertex(2)->point(), c->vertex(3)->point());
        const double r = std::sqrt(CGAL::squared_distance(center, c->vertex(0)->point()));
        for (int k = 0; k < 3; ++k)
            if (center[k] - r <= opt.box_lo[k] - opt.ghost_margin || center[k] + r >= opt.box_hi[k] + opt.ghost_margin)
                return false;
        return true;
    }

    // count v's data / random neighbors by traversing incident cells with a
    // custom bfs (without internal tds marks, so it is safe in parallel).
    // ghosts count as their original points. return false if v's star cannot
    // be verified as exact (periodic mode).
    bool count_neighbors(Vertex_handle v, const std::uint8_t* is_data, NeighborScratch& s, std::int32_t& nd,
                         std::int32_t& nr) const {
        bool verified = true;
        nd = nr = 0;
        s.cells.clear();
        s.seen_cells.clear();
        s.seen_points.clear();
        s.cells.push_back(v->cell());
        s.seen_cells.insert(reinterpret_cast<std::uintptr_t>(&*v->cell()));
        const Index self = v->info().id;
        for (std::size_t h = 0; h < s.cells.size(); ++h) {
            const Cell_handle c = s.cells[h];
            if (opt.periodic && verified) verified = !dt.is_infinite(c) && circumsphere_inside(c);
            const int iv = c->index(v);
            for (int k = 0; k < 4; ++k) {
                if (k == iv) continue;
                const Vertex_handle w = c->vertex(k);
                if (!dt.is_infinite(w)) {
                    const Index j = orig(w->info().id);
                    if (j != self && s.seen_points.insert(std::uintptr_t(j) + 1)) ++(is_data[j] ? nd : nr);
                }
                // the cell opposite vertex k also contains v
                const Cell_handle next = c->neighbor(k);
                if (s.seen_cells.insert(reinterpret_cast<std::uintptr_t>(&*next))) s.cells.push_back(next);
            }
        }
        return verified;
    }
};

Delaunay3::Delaunay3(const double* xyz, Index n, const DelaunayOptions& options)
    : impl_(std::make_unique<Impl>(xyz, n, options)) {}

Delaunay3::~Delaunay3() = default;

Index Delaunay3::n_points() const { return impl_->n; }
Index Delaunay3::n_ghosts() const { return Index(impl_->ghost_orig.size()); }
int Delaunay3::lock_grid_cells() const { return impl_->grid_cells; }
bool Delaunay3::parallel() const { return impl_->parallel; }
std::size_t Delaunay3::number_of_vertices() const { return impl_->dt.number_of_vertices(); }
std::size_t Delaunay3::number_of_finite_cells() const { return impl_->dt.number_of_finite_cells(); }
std::size_t Delaunay3::number_of_finite_edges() const { return impl_->dt.number_of_finite_edges(); }
bool Delaunay3::is_valid() const { return impl_->dt.is_valid(); }
const std::vector<std::array<Index, 2>>& Delaunay3::duplicates() const { return impl_->duplicates; }

NeighborCounts Delaunay3::neighbor_counts(const std::uint8_t* is_data, int threads) const {
    const DT& dt = impl_->dt;
    if (dt.dimension() < 3) throw std::runtime_error("all points are coplanar");
    const Index n = impl_->n;
    NeighborCounts out;
    out.ndata.assign(std::size_t(n), 0);
    out.nrand.assign(std::size_t(n), 0);

    // only vertices of original points (ghosts are not classified).
    std::vector<Vertex_handle> verts;
    verts.reserve(std::size_t(n));
    for (auto v = dt.finite_vertices_begin(); v != dt.finite_vertices_end(); ++v)
        if (v->info().id < n) verts.push_back(v);

    Index unverified = 0;
#pragma omp parallel num_threads(resolve_threads(threads)) reduction(+ : unverified)
    {
        NeighborScratch scratch;
        scratch.cells.reserve(128);
#pragma omp for schedule(dynamic, 4096)
        for (std::size_t k = 0; k < verts.size(); ++k) {
            const auto i = std::size_t(verts[k]->info().id);
            if (!impl_->count_neighbors(verts[k], is_data, scratch, out.ndata[i], out.nrand[i])) ++unverified;
        }
    }
    out.n_unverified = unverified;
    for (const auto& d : impl_->duplicates) {
        if (d[0] >= n) continue;  // duplicate ghost: do not classify
        const auto src = std::size_t(impl_->orig(d[1]));
        out.ndata[std::size_t(d[0])] = out.ndata[src];
        out.nrand[std::size_t(d[0])] = out.nrand[src];
    }
    return out;
}

std::vector<Index> Delaunay3::edges() const {
    const DT& dt = impl_->dt;
    const Index n = impl_->n;
    std::vector<std::array<Index, 2>> pairs;
    for (auto e = dt.finite_edges_begin(); e != dt.finite_edges_end(); ++e) {
        const Index ia = e->first->vertex(e->second)->info().id;
        const Index ib = e->first->vertex(e->third)->info().id;
        if (ia >= n && ib >= n) continue;  // edge between two ghosts
        Index a = impl_->orig(ia), b = impl_->orig(ib);
        if (a == b) continue;
        if (a > b) std::swap(a, b);
        pairs.push_back({a, b});
    }
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    std::vector<Index> out;
    out.reserve(2 * pairs.size());
    for (const auto& p : pairs) {
        out.push_back(p[0]);
        out.push_back(p[1]);
    }
    return out;
}

}  // namespace astra

// 3d delaunay triangulation with cgal and astra neighbor counts.
//
// each vertex stores its point's original index, so ndata/nrand counts align
// with the input array without materializing edges. cgal types remain hidden
// behind an opaque pointer (pimpl).
//
// periodic mode (cubic boxes): add ghost copies of points within
// `ghost_margin` of each face and count ghost neighbors as their original
// points. while counting, verify that each point's star matches the exact
// periodic triangulation (all circumspheres fit in the expanded box);
// report failures in `n_unverified`.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "astra/common.hpp"

namespace astra {

struct DelaunayOptions {
    int threads = 0;          // <= 1: sequential insertion; > 1: parallel (requires tbb)
    // lock grid resolution per axis (parallel mode). <= 0: automatic,
    // proportional to n^(1/3) so each lock cell covers a similar number
    // of points; large cells cause thread contention and stall insertion.
    int lock_grid_cells = 0;
    bool periodic = false;    // periodic box [box_lo, box_hi)
    std::array<double, 3> box_lo{}, box_hi{};
    double ghost_margin = 0;  // ghost copy layer width (periodic mode)
};

// automatic lock grid resolution for n points.
int auto_lock_grid_cells(Index n);

struct NeighborCounts {
    std::vector<std::int32_t> ndata;
    std::vector<std::int32_t> nrand;
    Index n_unverified = 0;   // periodic mode: points whose star could not be verified
};

class Delaunay3 {
public:
    // xyz: n contiguous (x, y, z) points as doubles. read only during
    // construction (cgal stores its own copy of each point).
    Delaunay3(const double* xyz, Index n, const DelaunayOptions& options);
    ~Delaunay3();
    Delaunay3(const Delaunay3&) = delete;
    Delaunay3& operator=(const Delaunay3&) = delete;

    Index n_points() const;
    Index n_ghosts() const;
    int lock_grid_cells() const;  // 0 if insertion was sequential
    bool parallel() const;
    std::size_t number_of_vertices() const;
    std::size_t number_of_finite_cells() const;
    std::size_t number_of_finite_edges() const;
    bool is_valid() const;

    // points identical to an already inserted point: (point, canonical vertex).
    // inherit counts from the canonical point.
    const std::vector<std::array<Index, 2>>& duplicates() const;

    // ndata and nrand per point: finite delaunay neighbors that are data / random.
    NeighborCounts neighbor_counts(const std::uint8_t* is_data, int threads) const;

    // edges (i, j) with i < j in original indices, unique, sorted, and flattened
    // (for validation on small samples). periodic mode includes edges crossing
    // the box.
    std::vector<Index> edges() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace astra

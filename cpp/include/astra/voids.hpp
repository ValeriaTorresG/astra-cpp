// astra void finder (forero/astra-desi, branch void-finder, module group_finder):
//
//   1. r = (ndata - nrand) / (ndata + nrand) for every delaunay vertex (data + randoms);
//   2. select points with r <= r_threshold and process them by ascending r
//      (ties by point index);
//   3. watershed: a point without an already assigned neighbor seeds a new group;
//      otherwise it joins the lowest-id neighboring group (basins never merge);
//   4. discard groups with fewer than min_members points (data + randoms);
//   5. cut-sky: angular (healpix) and radial selection from the randoms of the
//      realization. members in invalid regions and internal edges crossing them are
//      removed, each group keeps the connected component of its seed, and every
//      affected group is flagged as border;
//   6. shape of each group from its random members only: center = mean position,
//      moment tensor eigenvalues l1 >= l2 >= l3,
//          r_eff = sqrt(5) (l1 l2 l3)^(1/6),  ellip = 1 - ((l3 + l2) / (l2 + l1))^(1/4).
//
// everything runs in "rank space": the selected points are numbered by their
// position in the processing order, so the watershed reads its inputs as a stream.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "astra/common.hpp"
#include "astra/delaunay.hpp"

namespace astra {

// border causes, bits of VoidFinder::border()
enum BorderFlag : std::uint8_t {
    kBorderAngularMember = 1,  // a member falls in an invalid healpix pixel
    kBorderRadialMember = 2,   // a member falls in an invalid radial bin
    kBorderAngularEdge = 4,    // an internal edge crosses an invalid pixel
    kBorderRadialEdge = 8,     // an internal edge crosses an invalid radial bin
    kBorderBoxFace = 16,       // a member lies within the buffer of a face (nonperiodic box)
};

// flags per point in VoidFinder::membership
enum MemberFlag : std::uint8_t {
    kMemberSelected = 1,  // r <= r_threshold
    kMemberPruned = 2,    // in a group before the mask, removed by it
    kMemberBorder = 4,    // its group before the mask is a border group
};

struct Box {
    std::array<double, 3> lo{}, hi{};
};

// angular and radial selection counted from the randoms of one realization:
// a healpix pixel (ring) is valid with >= min_randoms_per_pixel randoms, and a
// radial bin [k w, (k + 1) w) with >= min_randoms_per_radial_bin randoms.
struct SelectionMask {
    std::int64_t nside = 128;
    int min_randoms_per_pixel = 3;
    int min_randoms_per_radial_bin = 3;
    double radial_bin_width = 5.0;
    std::vector<std::int64_t> pixel_counts;   // 12 nside^2
    std::vector<double> radial_edges;         // n_bins + 1
    std::vector<std::int64_t> radial_counts;  // n_bins
    std::vector<std::int64_t> invalid_prefix; // n_bins + 1: invalid bins before each bin
    Index n_rows = 0, n_valid_angular = 0, n_valid_radial = 0;

    bool pixel_valid(std::int64_t pixel) const { return pixel_counts[std::size_t(pixel)] >= min_randoms_per_pixel; }
    bool bin_valid(Index bin) const { return radial_counts[std::size_t(bin)] >= min_randoms_per_radial_bin; }
    Index n_bins() const { return Index(radial_counts.size()); }
    // bin of a radius within [edges[0], edges[-1]] (searchsorted right - 1, the
    // last edge belongs to the last bin)
    Index radial_bin(double radius) const;
    Index n_valid_pixels() const;
    Index n_valid_bins() const;
};

// mask from n randoms: (ra, dec) in degrees for the pixels and cartesian xyz for the radii.
SelectionMask build_selection_mask(const double* ra, const double* dec, const double* xyz, Index n,
                                   std::int64_t nside, int min_randoms_per_pixel, double radial_bin_width,
                                   int min_randoms_per_radial_bin, int threads);

struct MaskStats {
    Index n_internal_edges = 0;
    Index n_angular_edge_crossings = 0, n_radial_edge_crossings = 0;
    Index n_invalid_angular_members = 0, n_invalid_radial_members = 0;
    Index n_groups_affected = 0;      // border groups
    Index n_groups_pruned = 0;        // lost members (including discarded)
    Index n_groups_discarded = 0;     // lost all members
    Index n_groups_seed_invalid = 0;  // seed in an invalid region
    Index n_groups_undersized = 0;    // seed component with < min_members points
    Index n_pruned_members = 0, n_invalid_members_pruned = 0, n_disconnected_members = 0,
          n_small_component_members = 0;
};

// per group shapes, ordered by group id. undefined shapes (fewer than 3
// independent directions among the random members) are nan; the center is nan
// only for groups without random members.
struct VoidShapes {
    std::vector<std::int32_t> group_id;
    std::vector<std::int64_t> n_members, n_data, n_random;
    std::vector<double> center;  // 3 per group
    std::vector<double> eigval;  // 3 per group, descending
    std::vector<double> eigvec;  // 9 per group: row a is the unit eigenvector of eigval a
    std::vector<double> r_eff, ellip;
    // periodic boxes: 1 if the unwrapped random members span more than half the
    // box along some axis (the group may wrap around the box: shape ambiguous)
    std::vector<std::uint8_t> wrap_ambiguous;
};

class VoidFinder {
public:
    // selection and processing order from the delaunay neighbor counts of n points.
    VoidFinder(const std::int32_t* ndata, const std::int32_t* nrand, Index n, double r_threshold, int threads);

    Index n_points() const { return n_; }
    Index n_selected() const { return Index(order_.size()); }
    double r_threshold() const { return r_threshold_; }
    const std::vector<Index>& order() const { return order_; }          // rank -> point
    const std::vector<std::int32_t>& rank() const { return rank_; }     // point -> rank (-1 if not selected)

    // exact duplicate points (point, canonical point) have no delaunay vertex:
    // remove them from the watershed and give them the group of their canonical
    // point (same counts, hence the same r). call before set_adjacency.
    void merge_duplicates(const std::vector<std::array<Index, 2>>& duplicates);
    Index n_merged_duplicates() const { return Index(aliases_.size()); }

    // induced lower adjacency of the selected points (Delaunay3::lower_adjacency(rank())).
    void set_adjacency(LowerAdjacency&& adjacency);
    const LowerAdjacency& adjacency() const { return adj_; }
    void release_adjacency();

    // watershed + min_members cut. group ids are seed numbers in processing order.
    void grow(int min_members);
    Index n_seeds() const { return Index(seed_rank_.size()); }
    Index n_groups_premask() const { return n_groups_premask_; }
    Index n_groups() const;  // groups with members after the mask

    // cut-sky selection: prune members and edges in invalid regions (keeping
    // each group's seed component with >= min_members points) and flag border groups.
    MaskStats apply_selection_mask(const SelectionMask& mask, const double* xyz, const std::uint8_t* is_data,
                                   const double* ra, const double* dec, double angular_sample_step,
                                   int min_members, int threads);

    // nonperiodic box: flag groups with a member within `buffer` of a face. returns the number of groups.
    Index flag_box_border(const double* xyz, const Box& box, double buffer, int threads);

    // shapes of the final groups. coordinates are multiplied by `scale`
    // (e.g. h for mpc -> mpc/h). with `periodic`, members are unwrapped around
    // one of them and the center is wrapped back into the box.
    VoidShapes shapes(const double* xyz, const std::uint8_t* is_data, double scale, const Box* periodic,
                      int threads) const;

    // per point (n): group before the mask, final group, and MemberFlag bits.
    void membership(std::int32_t* premask, std::int32_t* group, std::uint8_t* flags, int threads) const;

    const std::vector<std::uint8_t>& border() const { return border_; }      // group id -> BorderFlag bits
    const std::vector<std::int64_t>& seed_rank() const { return seed_rank_; }  // group id -> rank of its seed
    const std::vector<std::int64_t>& seed_size() const { return seed_size_; }  // group id -> watershed size
    const std::vector<std::int32_t>& group_by_rank() const { return group_; }
    const std::vector<std::int32_t>& premask_by_rank() const { return premask_; }

private:
    Index n_ = 0;
    double r_threshold_ = 0;
    std::vector<Index> order_;
    std::vector<std::int32_t> rank_;
    LowerAdjacency adj_;
    bool grown_ = false;
    std::vector<std::int32_t> premask_;   // rank -> group after min_members (-1)
    std::vector<std::int32_t> group_;     // rank -> final group (-1)
    std::vector<std::int64_t> seed_rank_, seed_size_;
    std::vector<std::uint8_t> border_;
    std::vector<std::array<Index, 2>> aliases_;  // (merged duplicate, selected canonical point)
    Index n_groups_premask_ = 0;
};

// ---------------------------------------------------------------- consensus
// consensus catalog over voids pooled from several realizations (consensus.py):
//   2. visit the pooled spheres by decreasing radius; each unabsorbed seed groups
//      every sphere whose volume fraction inside the seed exceeds vol_frac;
//   3. each realization votes with its largest member and the representative is
//      the lower-median-radius vote, with support = number of voting realizations;
//   4. all group members are absorbed;
//   5. the representatives are pruned the same way, keeping the largest of each group.
// the support cut (support / n_runs > v_cut) is left to the caller.
struct ConsensusOptions {
    double vol_frac = 0.5;
    bool periodic = false;  // minimum-image distances in box
    Box box;
    Index batch = Index(1) << 15;  // seeds evaluated in parallel per batch
};

struct ConsensusResult {
    std::vector<Index> rows;              // pooled rows of the final representatives
    std::vector<std::int32_t> support;    // voting realizations
    std::vector<std::int32_t> n_absorbed; // representatives absorbed in pruning
    Index n_groups = 0;                   // representatives before pruning
    Index n_duplicates = 0;               // pruning picked an already kept representative (dropped)
};

ConsensusResult void_consensus(const double* centers, const double* radii, const std::int32_t* origins, Index n,
                               int n_runs, const ConsensusOptions& options, int threads);

// fraction of a sphere of radius `sphere` contained in a sphere of radius
// `container` whose center lies at `distance` (exact lens volume).
double contained_volume_fraction(double distance, double container, double sphere);

}  // namespace astra

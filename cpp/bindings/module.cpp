// pybind11 module `astra_cpp._core`: expose the c++ core as functions that
// receive and return numpy arrays without copies. all costly stages
// release the gil.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <CGAL/version.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "astra/box.hpp"
#include "astra/classify.hpp"
#include "astra/common.hpp"
#include "astra/cosmology.hpp"
#include "astra/delaunay.hpp"
#include "astra/fits.hpp"
#include "astra/healpix.hpp"
#include "astra/random_pool.hpp"
#include "astra/voids.hpp"

#include <algorithm>

namespace py = pybind11;
using namespace astra;

namespace {

template <class T>
using CArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

// hand ownership of std::vector memory to numpy (without copying).
template <class T>
py::array_t<T> to_numpy(std::vector<T>&& v, std::vector<py::ssize_t> shape) {
    auto* owner = new std::vector<T>(std::move(v));
    py::capsule capsule(owner, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    return py::array_t<T>(shape, owner->data(), capsule);
}

template <class T>
py::array_t<T> to_numpy(std::vector<T>&& v) {
    const auto n = static_cast<py::ssize_t>(v.size());
    return to_numpy(std::move(v), {n});
}

void require_1d(const py::array& a, py::ssize_t n, const char* name) {
    if (a.ndim() != 1 || a.shape(0) != n)
        throw std::invalid_argument(std::string(name) + " must have shape (" + std::to_string(n) + ",)");
}

const double* points_ptr(const CArray<double>& pts) {
    if (pts.ndim() != 2 || pts.shape(1) != 3) throw std::invalid_argument("points must have shape (N, 3)");
    return pts.data();
}

// output array written in place: require the exact dtype, contiguity, and
// writability to avoid writing into a temporary copy.
template <class T>
py::array_t<T> writable_array(const py::handle& obj, py::ssize_t ndim, const char* name) {
    if (!py::isinstance<py::array_t<T>>(obj))
        throw std::invalid_argument(std::string(name) + ": incorrect dtype");
    auto arr = py::reinterpret_borrow<py::array_t<T>>(obj);
    if (arr.ndim() != ndim || !(arr.flags() & py::array::c_style) || !arr.writeable())
        throw std::invalid_argument(std::string(name) + ": must be contiguous, writable, and " +
                                    std::to_string(ndim) + "-dimensional");
    return arr;
}

// writable 1d float64 column in place, contiguous or strided (e.g. xyz[:, 0]).
py::array_t<double> writable_column(const py::handle& obj, const char* name) {
    if (!py::isinstance<py::array_t<double>>(obj)) throw std::invalid_argument(std::string(name) + ": incorrect dtype");
    auto arr = py::reinterpret_borrow<py::array_t<double>>(obj);
    if (arr.ndim() != 1 || !arr.writeable() || arr.strides(0) <= 0 || arr.strides(0) % py::ssize_t(sizeof(double)))
        throw std::invalid_argument(std::string(name) + ": must be 1D, writable, and have a positive stride");
    return arr;
}

// contiguous writable (n, 3) float64 array in place.
double* writable_points(const py::handle& obj, const char* name, py::ssize_t& n) {
    auto arr = writable_array<double>(obj, 2, name);
    if (arr.shape(1) != 3) throw std::invalid_argument(std::string(name) + " must have shape (N, 3)");
    n = arr.shape(0);
    return arr.mutable_data();
}

py::array_t<double> new_points(Index n) { return py::array_t<double>({py::ssize_t(n), py::ssize_t(3)}); }

// ---------------------------------------------------------------- fits
void py_read_columns(const py::list& requests, const std::vector<std::string>& names, const py::list& outputs,
                     int threads, std::size_t block_mb) {
    std::vector<FitsReadRequest> reqs;
    std::vector<CArray<Index>> keep_rows;  // keep converted row arrays alive
    for (const auto& item : requests) {
        auto tup = item.cast<py::tuple>();
        if (tup.size() != 3) throw std::invalid_argument("each request must be (FitsTable, rows | None, out_offset)");
        FitsReadRequest r;
        r.table = &tup[0].cast<const FitsTable&>();
        r.out_offset = tup[2].cast<Index>();
        if (tup[1].is_none()) {
            r.all_rows = true;
            r.n_rows = r.table->nrows();
        } else {
            keep_rows.push_back(tup[1].cast<CArray<Index>>());
            const auto& rows = keep_rows.back();
            if (rows.ndim() != 1) throw std::invalid_argument("rows must be 1D");
            r.rows = rows.data();
            r.n_rows = rows.shape(0);
        }
        reqs.push_back(r);
    }
    if (outputs.size() != names.size()) throw std::invalid_argument("one output per column is required");
    std::vector<OutputColumn> outs;
    for (const auto& o : outputs) {
        auto arr = writable_column(o, "outputs");
        for (const auto& r : reqs)
            if (r.out_offset < 0 || r.out_offset + r.n_rows > arr.shape(0))
                throw std::out_of_range("the output is too short for the requested rows");
        outs.push_back({arr.mutable_data(), arr.strides(0) / py::ssize_t(sizeof(double))});
    }
    py::gil_scoped_release release;
    read_columns(reqs, names, outs, threads, block_mb << 20);
}

py::list py_select_rows(const py::list& tables, const std::string& name, int threads, std::size_t block_mb) {
    std::vector<const FitsTable*> ts;
    for (const auto& t : tables) ts.push_back(&t.cast<const FitsTable&>());
    std::vector<std::vector<Index>> rows;
    {
        py::gil_scoped_release release;
        rows = select_rows(ts, name, threads, block_mb << 20);
    }
    py::list out;
    for (auto& r : rows) out.append(to_numpy(std::move(r)));
    return out;
}

// ---------------------------------------------------------------- delaunay
class PyDelaunay3 {
public:
    PyDelaunay3(const CArray<double>& points, int threads, int lock_grid_cells,
                std::optional<std::array<double, 3>> box_lo, std::optional<std::array<double, 3>> box_hi,
                double ghost_margin) {
        const double* p = points_ptr(points);
        const Index n = points.shape(0);
        DelaunayOptions opt;
        opt.threads = threads;
        opt.lock_grid_cells = lock_grid_cells;
        if (box_lo.has_value() != box_hi.has_value())
            throw std::invalid_argument("box_lo and box_hi must be provided together");
        if (box_lo) {
            opt.periodic = true;
            opt.box_lo = *box_lo;
            opt.box_hi = *box_hi;
            opt.ghost_margin = ghost_margin;
        }
        py::gil_scoped_release release;
        dt_ = std::make_unique<Delaunay3>(p, n, opt);
    }

    py::tuple neighbor_counts(const CArray<bool>& is_data, int threads) const {
        require_1d(is_data, dt_->n_points(), "is_data");
        const auto* d = reinterpret_cast<const std::uint8_t*>(is_data.data());
        NeighborCounts c;
        {
            py::gil_scoped_release release;
            c = dt_->neighbor_counts(d, threads);
        }
        n_unverified_ = c.n_unverified;
        return py::make_tuple(to_numpy(std::move(c.ndata)), to_numpy(std::move(c.nrand)));
    }

    py::array_t<Index> edges() const {
        std::vector<Index> e;
        {
            py::gil_scoped_release release;
            e = dt_->edges();
        }
        const auto m = static_cast<py::ssize_t>(e.size() / 2);
        return to_numpy(std::move(e), {m, 2});
    }

    py::array_t<Index> duplicates() const {
        const auto& d = dt_->duplicates();
        py::array_t<Index> out({py::ssize_t(d.size()), py::ssize_t(2)});
        auto o = out.mutable_unchecked<2>();
        for (std::size_t k = 0; k < d.size(); ++k) { o(k, 0) = d[k][0]; o(k, 1) = d[k][1]; }
        return out;
    }

    const Delaunay3& get() const { return *dt_; }

    Index n_unverified() const { return n_unverified_; }

private:
    std::unique_ptr<Delaunay3> dt_;
    mutable Index n_unverified_ = -1;  // known after neighbor_counts
};

// ---------------------------------------------------------------- voids
template <class T>
py::array_t<T> copy_numpy(const std::vector<T>& v) {
    return py::array_t<T>(py::ssize_t(v.size()), v.data());
}

const std::uint8_t* bool_ptr(const CArray<bool>& a, py::ssize_t n, const char* name) {
    require_1d(a, n, name);
    return reinterpret_cast<const std::uint8_t*>(a.data());
}

std::optional<Box> make_box(const std::optional<std::array<double, 3>>& lo, const std::optional<std::array<double, 3>>& hi) {
    if (lo.has_value() != hi.has_value()) throw std::invalid_argument("box_lo and box_hi must be provided together");
    if (!lo) return std::nullopt;
    Box box{*lo, *hi};
    for (int c = 0; c < 3; ++c)
        if (!(box.hi[c] > box.lo[c])) throw std::invalid_argument("invalid box");
    return box;
}

class PyVoidFinder {
public:
    PyVoidFinder(const CArray<std::int32_t>& ndata, const CArray<std::int32_t>& nrand, double r_threshold, int threads) {
        const py::ssize_t n = ndata.size();
        require_1d(ndata, n, "ndata");
        require_1d(nrand, n, "nrand");
        py::gil_scoped_release release;
        vf_ = std::make_unique<VoidFinder>(ndata.data(), nrand.data(), n, r_threshold, threads);
    }

    std::int64_t build_adjacency(const PyDelaunay3& tri, int threads) {
        if (tri.get().n_points() != vf_->n_points())
            throw std::invalid_argument("the triangulation and the void finder have different points");
        py::gil_scoped_release release;
        const Delaunay3& dt = tri.get();
        std::vector<std::array<Index, 2>> duplicates;
        for (const auto& d : dt.duplicates())
            if (d[0] < dt.n_points()) duplicates.push_back({d[0], dt.original_point(d[1])});
        vf_->merge_duplicates(duplicates);
        vf_->set_adjacency(dt.lower_adjacency(vf_->rank().data(), vf_->n_selected(), threads));
        return vf_->adjacency().offsets.back();
    }

    py::tuple adjacency() const {
        const auto& a = vf_->adjacency();
        return py::make_tuple(copy_numpy(a.offsets), copy_numpy(a.neighbors));
    }

    void grow(int min_members) {
        py::gil_scoped_release release;
        vf_->grow(min_members);
    }

    py::dict apply_selection_mask(const SelectionMask& mask, const CArray<double>& xyz, const CArray<bool>& is_data,
                                  std::optional<CArray<double>> ra, std::optional<CArray<double>> dec,
                                  double angular_sample_step, int min_members, int threads) {
        const py::ssize_t n = vf_->n_points();
        if (xyz.ndim() != 2 || xyz.shape(0) != n || xyz.shape(1) != 3) throw std::invalid_argument("xyz must have shape (N, 3)");
        const auto* d = bool_ptr(is_data, n, "is_data");
        if (ra.has_value() != dec.has_value()) throw std::invalid_argument("ra and dec must be provided together");
        if (ra) { require_1d(*ra, n, "ra"); require_1d(*dec, n, "dec"); }
        const double* rp = ra ? ra->data() : nullptr;
        const double* dp = dec ? dec->data() : nullptr;
        MaskStats s;
        {
            py::gil_scoped_release release;
            s = vf_->apply_selection_mask(mask, xyz.data(), d, rp, dp, angular_sample_step, min_members, threads);
        }
        py::dict out;
        out["n_internal_edges"] = s.n_internal_edges;
        out["n_angular_edge_crossings"] = s.n_angular_edge_crossings;
        out["n_radial_edge_crossings"] = s.n_radial_edge_crossings;
        out["n_invalid_angular_members"] = s.n_invalid_angular_members;
        out["n_invalid_radial_members"] = s.n_invalid_radial_members;
        out["n_groups_affected"] = s.n_groups_affected;
        out["n_groups_pruned"] = s.n_groups_pruned;
        out["n_groups_discarded"] = s.n_groups_discarded;
        out["n_groups_seed_invalid"] = s.n_groups_seed_invalid;
        out["n_groups_undersized"] = s.n_groups_undersized;
        out["n_pruned_members"] = s.n_pruned_members;
        out["n_invalid_members_pruned"] = s.n_invalid_members_pruned;
        out["n_disconnected_members"] = s.n_disconnected_members;
        out["n_small_component_members"] = s.n_small_component_members;
        return out;
    }

    Index flag_box_border(const CArray<double>& xyz, std::array<double, 3> lo, std::array<double, 3> hi,
                          double buffer, int threads) {
        const py::ssize_t n = vf_->n_points();
        if (xyz.ndim() != 2 || xyz.shape(0) != n || xyz.shape(1) != 3) throw std::invalid_argument("xyz must have shape (N, 3)");
        const Box box = *make_box(lo, hi);
        py::gil_scoped_release release;
        return vf_->flag_box_border(xyz.data(), box, buffer, threads);
    }

    py::dict shapes(const CArray<double>& xyz, const CArray<bool>& is_data, double scale,
                    std::optional<std::array<double, 3>> box_lo, std::optional<std::array<double, 3>> box_hi,
                    int threads) const {
        const py::ssize_t n = vf_->n_points();
        if (xyz.ndim() != 2 || xyz.shape(0) != n || xyz.shape(1) != 3) throw std::invalid_argument("xyz must have shape (N, 3)");
        const auto* d = bool_ptr(is_data, n, "is_data");
        const auto box = make_box(box_lo, box_hi);
        VoidShapes s;
        {
            py::gil_scoped_release release;
            s = vf_->shapes(xyz.data(), d, scale, box ? &*box : nullptr, threads);
        }
        const auto rows = py::ssize_t(s.group_id.size());
        py::dict out;
        out["group_id"] = to_numpy(std::move(s.group_id));
        out["n_members"] = to_numpy(std::move(s.n_members));
        out["n_data"] = to_numpy(std::move(s.n_data));
        out["n_random"] = to_numpy(std::move(s.n_random));
        out["center"] = to_numpy(std::move(s.center), {rows, 3});
        out["eigval"] = to_numpy(std::move(s.eigval), {rows, 3});
        out["eigvec"] = to_numpy(std::move(s.eigvec), {rows, 3, 3});
        out["r_eff"] = to_numpy(std::move(s.r_eff));
        out["ellip"] = to_numpy(std::move(s.ellip));
        out["wrap_ambiguous"] = to_numpy(std::move(s.wrap_ambiguous));
        return out;
    }

    py::tuple membership(int threads) const {
        const auto n = py::ssize_t(vf_->n_points());
        py::array_t<std::int32_t> premask(n), group(n);
        py::array_t<std::uint8_t> flags(n);
        std::int32_t* pp = premask.mutable_data();
        std::int32_t* gp = group.mutable_data();
        std::uint8_t* fp = flags.mutable_data();
        {
            py::gil_scoped_release release;
            vf_->membership(pp, gp, fp, threads);
        }
        return py::make_tuple(premask, group, flags);
    }

    VoidFinder& get() { return *vf_; }
    const VoidFinder& get() const { return *vf_; }

private:
    std::unique_ptr<VoidFinder> vf_;
};

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Núcleo C++ de ASTRA: lectura FITS, cosmología, pool de randoms, Delaunay CGAL y clasificación";
    m.attr("CGAL_VERSION") = CGAL_VERSION_STR;
    m.attr("HAS_PARALLEL_TRIANGULATION") = has_parallel_triangulation();
    m.def("max_threads", [] { return resolve_threads(0); });
    m.def("auto_lock_grid_cells", &auto_lock_grid_cells, py::arg("n"));

    // fits
    py::class_<FitsTable>(m, "FitsTable")
        .def(py::init<std::string, int>(), py::arg("path"), py::arg("hdu") = 1)
        .def_property_readonly("path", &FitsTable::path)
        .def_property_readonly("nrows", &FitsTable::nrows)
        .def_property_readonly("row_bytes", &FitsTable::row_bytes)
        .def_property_readonly("columns", [](const FitsTable& t) {
            std::vector<std::pair<std::string, std::string>> out;
            for (const auto& c : t.columns()) out.emplace_back(c.name, std::to_string(c.repeat) + c.type);
            return out;
        })
        .def("has_column", &FitsTable::has_column, py::arg("name"));
    m.def("read_columns", &py_read_columns, py::arg("requests"), py::arg("names"), py::arg("outputs"),
          py::arg("threads") = 0, py::arg("block_mb") = 64,
          "Lee columnas numéricas de varias tablas FITS como float64.\n"
          "requests: lista de (FitsTable, rows int64 crecientes | None, out_offset).");
    m.def("select_rows", &py_select_rows, py::arg("tables"), py::arg("name"), py::arg("threads") = 0,
          py::arg("block_mb") = 64,
          "Filas (int64 crecientes) de cada tabla donde la columna entera o lógica name es distinta de cero.");

    // cosmology
    py::class_<FlatLambdaCDM>(m, "FlatLambdaCDM")
        .def(py::init<double, double, double, double, std::vector<double>>(), py::arg("H0"), py::arg("Om0"),
             py::arg("Tcmb0"), py::arg("Neff"), py::arg("m_nu"))
        .def_static("planck18", &FlatLambdaCDM::planck18)
        .def_property_readonly("H0", &FlatLambdaCDM::H0)
        .def_property_readonly("Om0", &FlatLambdaCDM::Om0)
        .def_property_readonly("Ode0", &FlatLambdaCDM::Ode0)
        .def_property_readonly("Ogamma0", &FlatLambdaCDM::Ogamma0)
        .def_property_readonly("Onu0", &FlatLambdaCDM::Onu0)
        .def_property_readonly("hubble_distance", &FlatLambdaCDM::hubble_distance)
        .def("inv_efunc", py::vectorize(&FlatLambdaCDM::inv_efunc), py::arg("z"))
        .def("comoving_distance", py::vectorize(&FlatLambdaCDM::comoving_distance), py::arg("z"));
    m.def("sky_to_cartesian", [](const CArray<double>& ra, const CArray<double>& dec, const CArray<double>& z,
                                 const FlatLambdaCDM& cosmo, int threads) {
        const py::ssize_t n = ra.size();
        require_1d(ra, n, "ra"); require_1d(dec, n, "dec"); require_1d(z, n, "z");
        auto out = new_points(n);
        double* o = out.mutable_data();
        py::gil_scoped_release release;
        sky_to_cartesian(ra.data(), dec.data(), z.data(), std::size_t(n), cosmo, o, threads);
        return out;
    }, py::arg("ra"), py::arg("dec"), py::arg("z"), py::arg("cosmo"), py::arg("threads") = 0,
       "(RA, DEC) en grados y z -> coordenadas comóviles (N, 3) en Mpc");

    // randoms
    m.def("sample_pool", [](const std::vector<Index>& sizes, Index n, std::uint64_t seed) {
        std::vector<std::vector<Index>> rows;
        {
            py::gil_scoped_release release;
            rows = sample_pool(sizes, n, seed);
        }
        py::list out;
        for (auto& r : rows) out.append(to_numpy(std::move(r)));
        return out;
    }, py::arg("sizes"), py::arg("n"), py::arg("seed"),
       "Muestra n filas sin reemplazo del pool unido; devuelve las filas locales de cada fuente");
    m.def("uniform_box", [](Index n, std::array<double, 3> lo, std::array<double, 3> hi, std::uint64_t seed,
                            int threads) {
        auto out = new_points(n);
        double* o = out.mutable_data();
        py::gil_scoped_release release;
        uniform_box(n, lo.data(), hi.data(), seed, o, threads);
        return out;
    }, py::arg("n"), py::arg("lo"), py::arg("hi"), py::arg("seed"), py::arg("threads") = 0);
    m.def("fill_uniform_box", [](const py::object& out, std::array<double, 3> lo, std::array<double, 3> hi,
                                 std::uint64_t seed, int threads) {
        py::ssize_t n = 0;
        double* o = writable_points(out, "out", n);
        py::gil_scoped_release release;
        uniform_box(n, lo.data(), hi.data(), seed, o, threads);
    }, py::arg("out"), py::arg("lo"), py::arg("hi"), py::arg("seed"), py::arg("threads") = 0,
       "Igual que uniform_box pero escribe en sitio sobre out (N, 3)");
    m.def("wrap_periodic", [](const py::object& points, std::array<double, 3> lo, std::array<double, 3> hi,
                              int threads) {
        py::ssize_t n = 0;
        double* p = writable_points(points, "points", n);
        py::gil_scoped_release release;
        wrap_periodic(p, n, lo, hi, threads);
    }, py::arg("points"), py::arg("lo"), py::arg("hi"), py::arg("threads") = 0,
       "Lleva en sitio cada punto de points (N, 3) a la caja [lo, hi) por periodicidad");

    // delaunay
    py::class_<PyDelaunay3>(m, "Delaunay3")
        .def(py::init<const CArray<double>&, int, int, std::optional<std::array<double, 3>>,
                      std::optional<std::array<double, 3>>, double>(),
             py::arg("points"), py::arg("threads") = 0, py::arg("lock_grid_cells") = 0,
             py::arg("box_lo") = py::none(), py::arg("box_hi") = py::none(), py::arg("ghost_margin") = 0.0,
             "Triangulación de Delaunay 3D. Con box_lo/box_hi la caja es periódica: se agregan\n"
             "copias fantasma a menos de ghost_margin de cada cara.")
        .def_property_readonly("n_ghosts", [](const PyDelaunay3& d) { return d.get().n_ghosts(); })
        .def_property_readonly("lock_grid_cells", [](const PyDelaunay3& d) { return d.get().lock_grid_cells(); })
        .def_property_readonly("n_unverified", &PyDelaunay3::n_unverified,
                               "Modo periódico: puntos cuya estrella no se pudo verificar (tras neighbor_counts)")
        .def("neighbor_counts", &PyDelaunay3::neighbor_counts, py::arg("is_data"), py::arg("threads") = 0,
             "Devuelve (NDATA, NRAND) int32 alineados con los puntos de entrada")
        .def("edges", &PyDelaunay3::edges)
        .def_property_readonly("duplicates", &PyDelaunay3::duplicates)
        .def_property_readonly("n_points", [](const PyDelaunay3& d) { return d.get().n_points(); })
        .def_property_readonly("parallel", [](const PyDelaunay3& d) { return d.get().parallel(); })
        .def("number_of_vertices", [](const PyDelaunay3& d) { return d.get().number_of_vertices(); })
        .def("number_of_finite_cells", [](const PyDelaunay3& d) { return d.get().number_of_finite_cells(); })
        .def("number_of_finite_edges", [](const PyDelaunay3& d) {
            py::gil_scoped_release release;
            return d.get().number_of_finite_edges();
        })
        .def("is_valid", [](const PyDelaunay3& d) {
            py::gil_scoped_release release;
            return d.get().is_valid();
        });

    // classification
    m.def("classify", [](const CArray<std::int32_t>& ndata, const CArray<std::int32_t>& nrand, float r_lower,
                         float r_med, float r_upper, int threads) {
        const py::ssize_t n = ndata.size();
        require_1d(ndata, n, "ndata"); require_1d(nrand, n, "nrand");
        py::array_t<float> r(n);
        py::array_t<std::uint8_t> webtype(n);
        float* rp = r.mutable_data();
        std::uint8_t* wp = webtype.mutable_data();
        {
            py::gil_scoped_release release;
            classify(ndata.data(), nrand.data(), std::size_t(n), RThresholds{r_lower, r_med, r_upper}, rp, wp, threads);
        }
        return py::make_tuple(r, webtype);
    }, py::arg("ndata"), py::arg("nrand"), py::arg("r_lower") = -0.25f, py::arg("r_med") = 0.25f,
       py::arg("r_upper") = 0.65f, py::arg("threads") = 0, "Devuelve (r float32, WEBTYPE uint8)");
    m.def("accumulate_webtypes", [](const py::object& counts_obj, const CArray<std::uint8_t>& webtype, int threads) {
        auto counts = writable_array<std::uint16_t>(counts_obj, 2, "counts");
        const py::ssize_t n = webtype.size();
        if (counts.shape(0) != n || counts.shape(1) != kNumWebTypes)
            throw std::invalid_argument("counts must have shape (N, 4)");
        std::uint16_t* c = counts.mutable_data();
        py::gil_scoped_release release;
        accumulate_webtypes(webtype.data(), std::size_t(n), c, threads);
    }, py::arg("counts"), py::arg("webtype"), py::arg("threads") = 0);
    // void finder
    m.attr("BORDER_ANGULAR_MEMBER") = int(kBorderAngularMember);
    m.attr("BORDER_RADIAL_MEMBER") = int(kBorderRadialMember);
    m.attr("BORDER_ANGULAR_EDGE") = int(kBorderAngularEdge);
    m.attr("BORDER_RADIAL_EDGE") = int(kBorderRadialEdge);
    m.attr("BORDER_BOX_FACE") = int(kBorderBoxFace);
    m.attr("MEMBER_SELECTED") = int(kMemberSelected);
    m.attr("MEMBER_PRUNED") = int(kMemberPruned);
    m.attr("MEMBER_BORDER") = int(kMemberBorder);

    m.def("ang2pix_ring", py::vectorize([](std::int64_t nside, double theta, double phi) {
        return healpix::ang2pix_ring(nside, theta, phi);
    }), py::arg("nside"), py::arg("theta"), py::arg("phi"), "healpy.ang2pix (RING)");
    m.def("radec2pix_ring", py::vectorize([](std::int64_t nside, double ra, double dec) {
        return healpix::radec2pix_ring(nside, ra, dec);
    }), py::arg("nside"), py::arg("ra"), py::arg("dec"), "pixel RING de (RA, DEC) en grados");
    m.def("vec2pix_ring", py::vectorize([](std::int64_t nside, double x, double y, double z) {
        return healpix::vec2pix_ring(nside, x, y, z);
    }), py::arg("nside"), py::arg("x"), py::arg("y"), py::arg("z"), "healpy.vec2pix (RING)");
    m.def("contained_volume_fraction", py::vectorize(&contained_volume_fraction), py::arg("distance"),
          py::arg("container"), py::arg("sphere"));

    py::class_<SelectionMask>(m, "SelectionMask")
        .def_readonly("nside", &SelectionMask::nside)
        .def_readonly("min_randoms_per_pixel", &SelectionMask::min_randoms_per_pixel)
        .def_readonly("min_randoms_per_radial_bin", &SelectionMask::min_randoms_per_radial_bin)
        .def_readonly("radial_bin_width", &SelectionMask::radial_bin_width)
        .def_readonly("n_rows", &SelectionMask::n_rows)
        .def_readonly("n_valid_angular_rows", &SelectionMask::n_valid_angular)
        .def_readonly("n_valid_radial_rows", &SelectionMask::n_valid_radial)
        .def_property_readonly("pixel_counts", [](const SelectionMask& s) { return copy_numpy(s.pixel_counts); })
        .def_property_readonly("radial_edges", [](const SelectionMask& s) { return copy_numpy(s.radial_edges); })
        .def_property_readonly("radial_counts", [](const SelectionMask& s) { return copy_numpy(s.radial_counts); })
        .def_property_readonly("n_valid_pixels", &SelectionMask::n_valid_pixels)
        .def_property_readonly("n_valid_radial_bins", &SelectionMask::n_valid_bins)
        .def_property_readonly("n_nonempty_pixels", [](const SelectionMask& s) {
            return Index(std::count_if(s.pixel_counts.begin(), s.pixel_counts.end(), [](std::int64_t c) { return c > 0; }));
        });
    m.def("build_selection_mask", [](const CArray<double>& ra, const CArray<double>& dec, const CArray<double>& xyz,
                                     std::int64_t nside, int min_randoms_per_pixel, double radial_bin_width,
                                     int min_randoms_per_radial_bin, int threads) {
        const py::ssize_t n = ra.size();
        require_1d(ra, n, "ra");
        require_1d(dec, n, "dec");
        if (xyz.ndim() != 2 || xyz.shape(0) != n || xyz.shape(1) != 3) throw std::invalid_argument("xyz must have shape (N, 3)");
        py::gil_scoped_release release;
        return build_selection_mask(ra.data(), dec.data(), xyz.data(), n, nside, min_randoms_per_pixel,
                                    radial_bin_width, min_randoms_per_radial_bin, threads);
    }, py::arg("ra"), py::arg("dec"), py::arg("xyz"), py::arg("nside") = 128, py::arg("min_randoms_per_pixel") = 3,
       py::arg("radial_bin_width") = 5.0, py::arg("min_randoms_per_radial_bin") = 3, py::arg("threads") = 0,
       "Máscara angular (HEALPix RING) y radial contada con los randoms de una realización");

    py::class_<PyVoidFinder>(m, "VoidFinder")
        .def(py::init<const CArray<std::int32_t>&, const CArray<std::int32_t>&, double, int>(), py::arg("ndata"),
             py::arg("nrand"), py::arg("r_threshold") = -0.25, py::arg("threads") = 0,
             "Selecciona r <= r_threshold y ordena por r creciente (empates por índice)")
        .def_property_readonly("n_points", [](const PyVoidFinder& v) { return v.get().n_points(); })
        .def_property_readonly("n_selected", [](const PyVoidFinder& v) { return v.get().n_selected(); })
        .def_property_readonly("r_threshold", [](const PyVoidFinder& v) { return v.get().r_threshold(); })
        .def_property_readonly("order", [](const PyVoidFinder& v) { return copy_numpy(v.get().order()); },
                               "rank -> punto (copia)")
        .def_property_readonly("rank", [](const PyVoidFinder& v) { return copy_numpy(v.get().rank()); },
                               "punto -> rank o -1 (copia)")
        .def("count_selected", [](const PyVoidFinder& v, Index start, Index stop) {
            const auto& rank = v.get().rank();
            start = std::clamp<Index>(start, 0, Index(rank.size()));
            stop = std::clamp<Index>(stop, start, Index(rank.size()));
            return Index(std::count_if(rank.begin() + start, rank.begin() + stop, [](std::int32_t r) { return r >= 0; }));
        }, py::arg("start"), py::arg("stop"), "Puntos seleccionados en [start, stop)")
        .def("build_adjacency", &PyVoidFinder::build_adjacency, py::arg("triangulation"), py::arg("threads") = 0,
             "Fusiona los duplicados exactos con su punto canónico y construye la adyacencia inferior\n"
             "inducida por los puntos seleccionados; devuelve el número de aristas")
        .def("adjacency", &PyVoidFinder::adjacency, "(offsets int64, vecinos int32) en espacio de rank (copia)")
        .def("release_adjacency", [](PyVoidFinder& v) { v.get().release_adjacency(); })
        .def("grow", &PyVoidFinder::grow, py::arg("min_members") = 4, "Watershed de ASTRA + corte min_members")
        .def_property_readonly("n_seeds", [](const PyVoidFinder& v) { return v.get().n_seeds(); })
        .def_property_readonly("n_merged_duplicates", [](const PyVoidFinder& v) { return v.get().n_merged_duplicates(); })
        .def_property_readonly("n_groups_premask", [](const PyVoidFinder& v) { return v.get().n_groups_premask(); })
        .def_property_readonly("n_groups", [](const PyVoidFinder& v) { return v.get().n_groups(); })
        .def("apply_selection_mask", &PyVoidFinder::apply_selection_mask, py::arg("mask"), py::arg("xyz"),
             py::arg("is_data"), py::arg("ra") = py::none(), py::arg("dec") = py::none(),
             py::arg("angular_sample_step") = 0.0, py::arg("min_members") = 4, py::arg("threads") = 0)
        .def("flag_box_border", &PyVoidFinder::flag_box_border, py::arg("xyz"), py::arg("box_lo"), py::arg("box_hi"),
             py::arg("buffer"), py::arg("threads") = 0)
        .def("shapes", &PyVoidFinder::shapes, py::arg("xyz"), py::arg("is_data"), py::arg("scale") = 1.0,
             py::arg("box_lo") = py::none(), py::arg("box_hi") = py::none(), py::arg("threads") = 0)
        .def("membership", &PyVoidFinder::membership, py::arg("threads") = 0,
             "(GROUP_ID_PREMASK int32, GROUP_ID int32, flags uint8) por punto")
        .def_property_readonly("border", [](const PyVoidFinder& v) { return copy_numpy(v.get().border()); })
        .def_property_readonly("seed_rank", [](const PyVoidFinder& v) { return copy_numpy(v.get().seed_rank()); })
        .def_property_readonly("seed_size", [](const PyVoidFinder& v) { return copy_numpy(v.get().seed_size()); })
        .def_property_readonly("premask_by_rank", [](const PyVoidFinder& v) { return copy_numpy(v.get().premask_by_rank()); })
        .def_property_readonly("group_by_rank", [](const PyVoidFinder& v) { return copy_numpy(v.get().group_by_rank()); });

    m.def("void_consensus", [](const CArray<double>& centers, const CArray<double>& radii,
                               const CArray<std::int32_t>& origins, int n_runs, double vol_frac,
                               std::optional<std::array<double, 3>> box_lo, std::optional<std::array<double, 3>> box_hi,
                               Index batch, int threads) {
        const py::ssize_t n = radii.size();
        require_1d(radii, n, "radii");
        require_1d(origins, n, "origins");
        if (centers.ndim() != 2 || centers.shape(0) != n || centers.shape(1) != 3)
            throw std::invalid_argument("centers must have shape (N, 3)");
        ConsensusOptions opt;
        opt.vol_frac = vol_frac;
        opt.batch = batch;
        if (const auto box = make_box(box_lo, box_hi)) {
            opt.periodic = true;
            opt.box = *box;
        }
        ConsensusResult r;
        {
            py::gil_scoped_release release;
            r = void_consensus(centers.data(), radii.data(), origins.data(), n, n_runs, opt, threads);
        }
        py::dict out;
        out["rows"] = to_numpy(std::move(r.rows));
        out["support"] = to_numpy(std::move(r.support));
        out["n_absorbed"] = to_numpy(std::move(r.n_absorbed));
        out["n_groups"] = r.n_groups;
        out["n_duplicates"] = r.n_duplicates;
        return out;
    }, py::arg("centers"), py::arg("radii"), py::arg("origins"), py::arg("n_runs"), py::arg("vol_frac") = 0.5,
       py::arg("box_lo") = py::none(), py::arg("box_hi") = py::none(), py::arg("batch") = 32768,
       py::arg("threads") = 0,
       "Consenso de voids de varias realizaciones: filas representativas, soporte y absorbidos");

    m.def("counts_to_probabilities", [](const CArray<std::uint16_t>& counts, int n_iterations, int threads) {
        if (counts.ndim() != 2 || counts.shape(1) != kNumWebTypes) throw std::invalid_argument("counts must have shape (N, 4)");
        if (n_iterations <= 0) throw std::invalid_argument("n_iterations must be positive");
        const py::ssize_t n = counts.shape(0);
        py::array_t<float> prob({n, py::ssize_t(kNumWebTypes)});
        float* p = prob.mutable_data();
        py::gil_scoped_release release;
        counts_to_probabilities(counts.data(), std::size_t(n), n_iterations, p, threads);
        return prob;
    }, py::arg("counts"), py::arg("n_iterations"), py::arg("threads") = 0);
}

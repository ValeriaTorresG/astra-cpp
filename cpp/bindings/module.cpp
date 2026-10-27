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
#include "astra/random_pool.hpp"

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

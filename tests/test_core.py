"""tests for the c++ core (`astra_cpp._core`).

    python -m unittest discover -s tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from astra_cpp import _core  # noqa: E402

from fits_writer import write_bintable  # noqa: E402


def scipy_counts(points, is_data):
    """ndata/nrand as in astra-desi process_delaunay (scipy)."""
    from scipy.spatial import Delaunay
    indptr, indices = Delaunay(points).vertex_neighbor_vertices
    n = len(points)
    owner = np.repeat(np.arange(n), np.diff(indptr))
    ndata = np.bincount(owner[is_data[indices]], minlength=n)
    return ndata, np.diff(indptr) - ndata


def periodic_reference_counts(points, is_data, lo, hi):
    """exact periodic ndata/nrand: scipy over 27 copies of the box, counting
    neighbors of the central copy by original point."""
    import itertools
    from scipy.spatial import Delaunay
    n = len(points)
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3))) * (np.asarray(hi) - np.asarray(lo))
    indptr, indices = Delaunay(np.concatenate([points + s for s in shifts])).vertex_neighbor_vertices
    center = int(np.flatnonzero(~shifts.any(axis=1))[0])
    ndata = np.empty(n, dtype=np.int64)
    nrand = np.empty(n, dtype=np.int64)
    for i in range(n):
        v = center * n + i
        nbrs = np.unique(indices[indptr[v]:indptr[v + 1]] % n)
        nbrs = nbrs[nbrs != i]
        ndata[i] = is_data[nbrs].sum()
        nrand[i] = len(nbrs) - ndata[i]
    return ndata, nrand


def numpy_classify(ndata, nrand, lower=-0.25, med=0.25, upper=0.65):
    """r and classes as in implement_astra._compute_probability_components."""
    ndata, nrand = ndata.astype(np.float32), nrand.astype(np.float32)
    denom = ndata + nrand
    r = np.zeros_like(denom)
    np.divide(ndata - nrand, denom, out=r, where=denom > 0)
    cls = np.zeros(len(r), dtype=np.uint8)
    cls[(r >= lower) & (r < med)] = 1
    cls[(r >= med) & (r < upper)] = 2
    cls[r >= upper] = 3
    return r, cls


class TestDelaunay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.points = rng.random((50_000, 3)) * 500
        cls.is_data = rng.random(len(cls.points)) < 0.5
        cls.ref = scipy_counts(cls.points, cls.is_data)

    def test_counts_match_scipy(self):
        threads = [1, 4] if _core.HAS_PARALLEL_TRIANGULATION else [1]
        for th in threads:
            with self.subTest(threads=th):
                tri = _core.Delaunay3(self.points, threads=th)
                self.assertEqual(tri.parallel, th > 1)
                self.assertEqual(tri.number_of_vertices(), len(self.points))
                ndata, nrand = tri.neighbor_counts(self.is_data, threads=th)
                np.testing.assert_array_equal(ndata, self.ref[0])
                np.testing.assert_array_equal(nrand, self.ref[1])

    def test_edges_consistent_with_counts(self):
        tri = _core.Delaunay3(self.points, threads=1)
        edges = tri.edges()
        self.assertTrue(np.all(edges[:, 0] < edges[:, 1]))
        degree = np.bincount(edges.ravel(), minlength=len(self.points))
        np.testing.assert_array_equal(degree, self.ref[0] + self.ref[1])
        self.assertTrue(tri.is_valid())

    def test_duplicates_inherit_counts(self):
        pts = np.vstack([self.points[:2000], self.points[:5]])
        is_data = np.ones(len(pts), dtype=bool)
        tri = _core.Delaunay3(pts, threads=1)
        self.assertEqual(len(tri.duplicates), 5)
        ndata, nrand = tri.neighbor_counts(is_data)
        np.testing.assert_array_equal(ndata[2000:], ndata[:5])
        np.testing.assert_array_equal(nrand[2000:], nrand[:5])


class TestPeriodicDelaunay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(12)
        cls.lo, cls.hi = [-50.0] * 3, [50.0] * 3
        cls.points = rng.uniform(-50, 50, (8000, 3))
        cls.is_data = rng.random(len(cls.points)) < 0.5
        cls.ref = periodic_reference_counts(cls.points, cls.is_data, cls.lo, cls.hi)
        cls.spacing = (100.0 ** 3 / len(cls.points)) ** (1 / 3)

    def test_matches_tiled_scipy(self):
        threads = [1, 4] if _core.HAS_PARALLEL_TRIANGULATION else [1]
        for th in threads:
            with self.subTest(threads=th):
                tri = _core.Delaunay3(self.points, threads=th, box_lo=self.lo, box_hi=self.hi,
                                      ghost_margin=8 * self.spacing)
                self.assertGreater(tri.n_ghosts, 0)
                ndata, nrand = tri.neighbor_counts(self.is_data, threads=th)
                self.assertEqual(tri.n_unverified, 0)
                np.testing.assert_array_equal(ndata, self.ref[0])
                np.testing.assert_array_equal(nrand, self.ref[1])

    def test_periodic_edges(self):
        tri = _core.Delaunay3(self.points, threads=1, box_lo=self.lo, box_hi=self.hi,
                              ghost_margin=8 * self.spacing)
        degree = np.bincount(tri.edges().ravel(), minlength=len(self.points))
        np.testing.assert_array_equal(degree, self.ref[0] + self.ref[1])

    def test_small_margin_is_detected(self):
        tri = _core.Delaunay3(self.points, threads=1, box_lo=self.lo, box_hi=self.hi,
                              ghost_margin=0.5 * self.spacing)
        ndata, nrand = tri.neighbor_counts(self.is_data)
        wrong = (ndata != self.ref[0]) | (nrand != self.ref[1])
        self.assertGreater(tri.n_unverified, 0)
        self.assertGreaterEqual(tri.n_unverified, wrong.sum())


class TestClassify(unittest.TestCase):
    def test_matches_numpy(self):
        rng = np.random.default_rng(1)
        ndata = rng.integers(0, 30, 100_000).astype(np.int32)
        nrand = rng.integers(0, 30, 100_000).astype(np.int32)
        ndata[:10] = 0
        nrand[:10] = 0
        r, cls = _core.classify(ndata, nrand, -0.25, 0.25, 0.65)
        r_ref, cls_ref = numpy_classify(ndata, nrand)
        np.testing.assert_array_equal(r, r_ref)
        np.testing.assert_array_equal(cls, cls_ref)

    def test_thresholds_inclusive_on_the_left(self):
        # r exactly at the threshold falls in the upper class
        ndata = np.array([3, 5, 1, 0], dtype=np.int32)  # r = 0.5, 1, -1/3, 0 (no neighbors)
        nrand = np.array([1, 0, 2, 0], dtype=np.int32)
        _, cls = _core.classify(ndata, nrand, -0.5, 0.5, 1.0)
        np.testing.assert_array_equal(cls, [2, 3, 1, 1])

    def test_accumulate_and_probabilities(self):
        counts = np.zeros((4, 4), dtype=np.uint16)
        for w in ([0, 1, 2, 3], [0, 1, 2, 2], [3, 1, 2, 0]):
            _core.accumulate_webtypes(counts, np.array(w, dtype=np.uint8))
        prob = _core.counts_to_probabilities(counts, 3)
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, rtol=1e-6)
        np.testing.assert_allclose(prob[2], [0, 0, 1, 0])
        with self.assertRaises(ValueError):
            _core.accumulate_webtypes(np.zeros((4, 4), dtype=np.int64), np.zeros(4, dtype=np.uint8))


class TestRandomPool(unittest.TestCase):
    def test_sample_properties(self):
        sizes = [1000, 0, 2500, 777]
        for n in (0, 10, 1500, 4000, sum(sizes)):
            rows = _core.sample_pool(sizes, n, 42)
            self.assertEqual(sum(len(r) for r in rows), n)
            for r, size in zip(rows, sizes):
                self.assertTrue(np.all(np.diff(r) > 0))
                self.assertTrue(r.size == 0 or (r[0] >= 0 and r[-1] < size))

    def test_deterministic_and_seed_dependent(self):
        a = _core.sample_pool([10_000] * 3, 5000, 0)
        b = _core.sample_pool([10_000] * 3, 5000, 0)
        c = _core.sample_pool([10_000] * 3, 5000, 1)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        self.assertFalse(all(np.array_equal(x, y) for x, y in zip(a, c)))

    def test_uniform_over_pool(self):
        sizes = [200_000, 100_000, 300_000]
        rows = _core.sample_pool(sizes, 60_000, 3)
        frac = np.array([len(r) for r in rows]) / 60_000
        np.testing.assert_allclose(frac, np.array(sizes) / sum(sizes), atol=0.01)

    def test_too_many_rows(self):
        with self.assertRaises(ValueError):
            _core.sample_pool([10, 10], 21, 0)

    def test_fill_and_wrap_in_place(self):
        out = np.zeros((1000, 3))
        _core.fill_uniform_box(out, [0, 0, 0], [10, 10, 10], 4, threads=2)
        np.testing.assert_array_equal(out, _core.uniform_box(1000, [0, 0, 0], [10, 10, 10], 4))
        pts = np.array([[-1.0, 10.0, 25.0], [3.0, -10.0, 9.999]])
        _core.wrap_periodic(pts, [0, 0, 0], [10, 10, 10])
        np.testing.assert_allclose(pts, [[9, 0, 5], [3, 0, 9.999]])
        with self.assertRaises(ValueError):
            _core.wrap_periodic(np.zeros((4, 3))[:, ::2], [0] * 3, [1] * 3)

    def test_uniform_box(self):
        a = _core.uniform_box(10_000, [-5, 0, 10], [5, 1, 20], 9, threads=1)
        b = _core.uniform_box(10_000, [-5, 0, 10], [5, 1, 20], 9, threads=4)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(a.min(axis=0) >= [-5, 0, 10]) and np.all(a.max(axis=0) < [5, 1, 20]))


class TestCosmology(unittest.TestCase):
    # astropy.cosmology.Planck18.comoving_distance (astropy 7.2)
    Z = [0.01, 0.1, 0.2, 0.3, 0.5, 1.0]
    ASTROPY = [44.20505557661846, 432.5656977383524, 843.577899429573,
               1232.706667272353, 1946.4166347871185, 3395.6344711515662]

    def test_comoving_distance_matches_astropy(self):
        cosmo = _core.FlatLambdaCDM.planck18()
        np.testing.assert_allclose(cosmo.comoving_distance(np.array(self.Z)), self.ASTROPY, rtol=1e-10)

    def test_sky_to_cartesian(self):
        cosmo = _core.FlatLambdaCDM.planck18()
        rng = np.random.default_rng(0)
        ra, dec, z = rng.uniform(0, 360, 1000), rng.uniform(-90, 90, 1000), rng.uniform(0.01, 0.6, 1000)
        xyz = _core.sky_to_cartesian(ra, dec, z, cosmo, threads=2)
        d = cosmo.comoving_distance(z)
        np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), d, rtol=1e-12)
        np.testing.assert_allclose(xyz[:, 2], d * np.sin(np.deg2rad(dec)), rtol=1e-12, atol=1e-9)
        with self.assertRaises(RuntimeError):
            _core.sky_to_cartesian(ra, dec, np.where(np.arange(1000) == 3, -1.0, z), cosmo)


class TestFits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(5)
        n = 50_000
        cls.data = {
            'B': rng.integers(0, 255, n).astype(np.uint8),
            'I': rng.integers(-3000, 3000, n).astype(np.int16),
            'J': rng.integers(-10**9, 10**9, n).astype(np.int32),
            'K': rng.integers(-10**12, 10**12, n).astype(np.int64),
            'E': rng.random(n).astype(np.float32),
            'D': rng.random(n),
            'NAME': np.array([f'g{i:07d}' for i in range(n)], dtype='S8'),
            'FLAG': rng.random(n) < 0.3,
        }
        cls.path = Path(cls.tmp.name) / 'table.fits'
        write_bintable(cls.path, cls.data)
        cls.table = _core.FitsTable(str(cls.path))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_header(self):
        self.assertEqual(self.table.nrows, 50_000)
        self.assertEqual([c for c, _ in self.table.columns], list(self.data))
        self.assertTrue(self.table.has_column('d') and not self.table.has_column('X'))

    def read(self, rows, names=('B', 'I', 'J', 'K', 'E', 'D'), offset=0, threads=4):
        n = self.table.nrows if rows is None else len(rows)
        out = [np.full(n + offset, np.nan) for _ in names]
        _core.read_columns([(self.table, rows, offset)], list(names), out, threads=threads, block_mb=1)
        return [o[offset:] for o in out]

    def test_all_rows(self):
        for name, col in zip(('B', 'I', 'J', 'K', 'E', 'D'), self.read(None)):
            np.testing.assert_array_equal(col, self.data[name].astype(np.float64))

    def test_dense_and_sparse_subsets(self):
        dense = np.arange(100, 40_000, 2, dtype=np.int64)
        sparse = np.array([3, 15_000, 15_001, 49_999], dtype=np.int64)
        for rows in (dense, sparse):
            for name, col in zip(('J', 'D'), self.read(rows, ('J', 'D'), offset=7)):
                np.testing.assert_array_equal(col, self.data[name][rows].astype(np.float64))

    def test_strided_output(self):
        xyz = np.zeros((self.table.nrows + 3, 3))
        _core.read_columns([(self.table, None, 3)], ['E', 'J', 'D'], [xyz[:, 0], xyz[:, 1], xyz[:, 2]])
        for k, name in enumerate(('E', 'J', 'D')):
            np.testing.assert_array_equal(xyz[3:, k], self.data[name].astype(np.float64))
        np.testing.assert_array_equal(xyz[:3], 0)

    def test_rejects_bad_input(self):
        with self.assertRaises(RuntimeError):
            self.read(None, ('NAME',))
        with self.assertRaises(ValueError):
            self.read(np.array([5, 3], dtype=np.int64), ('D',))
        with self.assertRaises(ValueError):  # float32 output: would write into a copy
            _core.read_columns([(self.table, None, 0)], ['D'], [np.empty(50_000, np.float32)])

    def test_select_rows(self):
        for name in ('FLAG', 'B', 'J'):
            rows = _core.select_rows([self.table, self.table], name, threads=4, block_mb=1)
            self.assertEqual(len(rows), 2)
            for r in rows:
                self.assertEqual(r.dtype, np.int64)
                np.testing.assert_array_equal(r, np.flatnonzero(self.data[name]))
        with self.assertRaises(RuntimeError):
            _core.select_rows([self.table], 'D')


if __name__ == '__main__':
    unittest.main()

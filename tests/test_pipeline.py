"""end-to-end pipeline test on synthetic fits catalogs.

    python -m unittest discover -s tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from astra_cpp import CubicConfig, CutskyConfig, _core, run_cubic, run_cutsky  # noqa: E402
from astra_cpp.config import RANDOM_ID_FACTOR  # noqa: E402
from astra_cpp.outputs import CLASS_DTYPE, PROB_DTYPE, RunPaths  # noqa: E402

from fits_writer import write_bintable  # noqa: E402
from test_core import numpy_classify, periodic_reference_counts, scipy_counts  # noqa: E402


def sky_columns(rng, n):
    return {'RA': rng.uniform(150, 200, n), 'DEC': rng.uniform(0, 30, n),
            'Z': rng.uniform(0.05, 0.3, n).astype(np.float32),
            'Z_COSMO': rng.uniform(0.05, 0.3, n).astype(np.float32)}


class TestCutskyPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        rng = np.random.default_rng(11)
        cls.n_real = 3000
        write_bintable(root / 'real.fits', sky_columns(rng, cls.n_real))
        cls.random_sizes = {100: 2500, 200: 4000}
        for seed, n in cls.random_sizes.items():
            write_bintable(root / f'random_S{seed}_1X.fits', sky_columns(rng, n))
        cls.cfg = dict(real_file=root / 'real.fits', random_glob=str(root / 'random_S*_1X.fits'),
                       output_dir=root / 'out', threads=2, log_interval=3600)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_iterations_and_aggregate(self):
        for it in (0, 1):
            summary = run_cutsky(CutskyConfig(iteration=it, **self.cfg))
            self.assertEqual(summary['status'], 'complete')
        paths = RunPaths(CutskyConfig(**self.cfg).run_dir)
        n = 2 * self.n_real
        points = np.load(paths.points(0))
        cls = np.load(paths.classification(0))
        self.assertEqual(cls.dtype, CLASS_DTYPE)
        self.assertEqual(len(cls), n)
        self.assertTrue(cls['ISDATA'][:self.n_real].all() and not cls['ISDATA'][self.n_real:].any())
        np.testing.assert_array_equal(cls['TARGETID'][:self.n_real], np.arange(1, self.n_real + 1))

        # randoms: targetid encodes (seed, row), and positions come from the correct file
        rid = -cls['TARGETID'][self.n_real:]
        seeds, rows = (rid - 1) // RANDOM_ID_FACTOR, (rid - 1) % RANDOM_ID_FACTOR
        self.assertEqual(set(np.unique(seeds)), set(self.random_sizes))
        self.assertEqual(len(np.unique(rid)), self.n_real)
        table = _core.FitsTable(str(Path(self.cfg['random_glob']).with_name('random_S200_1X.fits')))
        sel = seeds == 200
        ra = np.empty(sel.sum())
        _core.read_columns([(table, rows[sel].astype(np.int64), 0)], ['RA'], [ra])
        np.testing.assert_array_equal(points['RA'][self.n_real:][sel], ra.astype(np.float32))

        # same ndata/nrand/r/classes as the original astra with scipy
        xyz = _core.sky_to_cartesian(points['RA'].astype(np.float64), points['DEC'].astype(np.float64),
                                     points['Z'].astype(np.float64), _core.FlatLambdaCDM.planck18())
        ndata, nrand = scipy_counts(xyz, cls['ISDATA'])
        np.testing.assert_array_equal(cls['NDATA'], ndata)
        np.testing.assert_array_equal(cls['NRAND'], nrand)
        r, webtype = numpy_classify(ndata, nrand)
        np.testing.assert_array_equal(cls['R'], r)
        np.testing.assert_array_equal(cls['WEBTYPE'], webtype)

        # probability = fraction of iterations in each class
        prob = np.load(paths.probability())
        self.assertEqual(prob.dtype, PROB_DTYPE)
        c1 = np.load(paths.classification(1))['WEBTYPE'][:self.n_real]
        expected = (np.eye(4)[webtype[:self.n_real]] + np.eye(4)[c1]) / 2
        got = np.column_stack([prob[f] for f in ('PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT')])
        np.testing.assert_allclose(got, expected)

    def test_skips_completed_iteration(self):
        run_cutsky(CutskyConfig(iteration=5, **self.cfg))
        self.assertEqual(run_cutsky(CutskyConfig(iteration=5, **self.cfg))['status'], 'skipped')


class TestCubicPipeline(unittest.TestCase):
    def test_periodic_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(3)
            n = 3000
            xyz = rng.uniform(-50, 50, (n, 3)).astype(np.float32)
            xyz[0] = [50.0, -50.0, 0.0]          # at the upper boundary: wraps to -50
            path = Path(tmp) / 'box.fits'
            write_bintable(path, {'X': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2]})
            cfg = CubicConfig(box_file=path, box_size=100.0, output_dir=Path(tmp) / 'out', threads=2,
                              iteration=4, log_interval=3600)
            summary = run_cubic(cfg)
            self.assertTrue(summary['periodic'])
            self.assertEqual(summary['stats']['unverified'], 0)
            self.assertGreater(summary['stats']['ghosts'], 0)
            cls = np.load(RunPaths(cfg.run_dir).classification(4))
            self.assertEqual(len(cls), 2 * n)
            np.testing.assert_array_equal(cls['TARGETID'][:n], np.arange(1, n + 1))
            np.testing.assert_array_equal(cls['TARGETID'][n:], -np.arange(1, n + 1))

            # reference: scipy over 27 copies with the same randoms (seed = iteration)
            data = xyz.astype(np.float64)
            data[0, 0] = -50.0
            points = np.vstack([data, _core.uniform_box(n, [-50] * 3, [50] * 3, 4)])
            ndata, nrand = periodic_reference_counts(points, cls['ISDATA'], [-50] * 3, [50] * 3)
            np.testing.assert_array_equal(cls['NDATA'], ndata)
            np.testing.assert_array_equal(cls['NRAND'], nrand)

    def test_box_with_uniform_randoms(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(2)
            xyz = rng.uniform(-50, 50, (4000, 3)).astype(np.float32)
            path = Path(tmp) / 'box.fits'
            write_bintable(path, {'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2]})
            cfg = CubicConfig(box_file=path, box_size=100.0, output_dir=Path(tmp) / 'out', threads=2,
                              subbox=60.0, log_interval=3600)
            summary = run_cubic(cfg)
            self.assertEqual(summary['box_lo'], [-50.0] * 3)
            points = np.load(RunPaths(cfg.run_dir).points(0))
            cls = np.load(RunPaths(cfg.run_dir).classification(0))
            n_data = summary['n_real']
            inside = np.all(xyz < 10.0, axis=1)
            self.assertEqual(n_data, inside.sum())
            np.testing.assert_array_equal(cls['TARGETID'][:n_data], np.flatnonzero(inside) + 1)
            self.assertTrue(np.all(points['XCART'] < 10.0))
            self.assertEqual(len(cls), 2 * n_data)


if __name__ == '__main__':
    unittest.main()

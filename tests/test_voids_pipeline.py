"""end-to-end tests of the void finder (`astra_cpp.voids`) on synthetic catalogs.

    python -m unittest discover -s tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from astra_cpp import CutskyConfig, _core, run_cutsky  # noqa: E402
from astra_cpp.catalogs import Catalog  # noqa: E402
from astra_cpp.monitor import ResourceMonitor, setup_logger  # noqa: E402
from astra_cpp.outputs import RunPaths  # noqa: E402
from astra_cpp.voids import VoidCubicConfig, VoidCutskyConfig, find_voids, run_cubic_voids, run_cutsky_voids  # noqa: E402
from astra_cpp.voids.cli import main as voids_main  # noqa: E402
from astra_cpp.voids.config import VOID_ID_FACTOR  # noqa: E402
from astra_cpp.voids.outputs import (CATALOG_DTYPE, CLEAN_DTYPE, CONSENSUS_DTYPE, MEMBERSHIP_DTYPE,  # noqa: E402
                                     VoidPaths)

from fits_writer import write_bintable  # noqa: E402
from test_voids import (cutsky_sample, ref_apply_mask, ref_graph, ref_group, ref_mask, ref_r,  # noqa: E402
                        ref_shapes)

LOG = setup_logger(name='astra_cpp.tests')


def clustered_sky(rng, n, centers):
    """(ra, dec, z) around a few cluster centers plus a uniform background."""
    k = rng.integers(0, len(centers), n)
    ra = centers[k, 0] + rng.normal(0, 0.8, n)
    dec = centers[k, 1] + rng.normal(0, 0.8, n)
    z = centers[k, 2] + rng.normal(0, 0.004, n)
    bg = rng.random(n) < 0.4
    ra[bg], dec[bg], z[bg] = rng.uniform(150, 200, bg.sum()), rng.uniform(0, 30, bg.sum()), rng.uniform(0.05, 0.3, bg.sum())
    keep = (ra > 150) & (ra < 200) & (dec > 0) & (dec < 30) & (z > 0.05) & (z < 0.3)
    return ra[keep], dec[keep], z[keep]


class TestFindVoids(unittest.TestCase):
    """find_voids on an in-memory cut-sky catalog against the reference port."""

    def test_cutsky_catalog_matches_reference(self):
        xyz, is_data, ra, dec = cutsky_sample(21, 12000)
        n = len(xyz)
        n_data = int(is_data.sum())
        targetid = np.where(is_data, np.arange(1, n + 1), -np.arange(1, n + 1))
        cat = Catalog(targetid, is_data, xyz.copy(), n_data, {'RA': ra, 'DEC': dec}, {'distance_unit': 'Mpc'})
        cfg = VoidCutskyConfig(threads=3, healpix_nside=64, iteration=7)
        with ResourceMonitor(LOG, 3600) as mon:
            all_voids, clean, members, stats = find_voids(cat, cfg, 7, LOG, mon)

        indptr, indices = ref_graph(xyz)
        r, _, _ = ref_r(indptr, indices, is_data)
        gid, order, _ = ref_group(indptr, indices, r)
        rnd = ~is_data
        mask = ref_mask(ra[rnd], dec[rnd], xyz[rnd], 64, 3, 5.0, 3)
        final, border, _ = ref_apply_mask(xyz, is_data, ra, dec, indptr, indices, gid, order, mask)
        np.testing.assert_array_equal(members['GROUP_ID_PREMASK'], gid)
        np.testing.assert_array_equal(members['GROUP_ID'], final)
        np.testing.assert_array_equal(members['SELECTED'], r <= -0.25)
        np.testing.assert_array_equal(members['R'], ((r.astype(np.float32))))
        np.testing.assert_array_equal(members['VOID_ID'], np.where(final >= 0, 7 * VOID_ID_FACTOR + final, -1))
        any_border = set().union(*border.values())
        self.assertEqual(set(all_voids['GROUP_ID'][all_voids['BORDER']].tolist()),
                         any_border & set(all_voids['GROUP_ID'].tolist()))
        h = _core.FlatLambdaCDM.planck18().H0 / 100
        self.assertAlmostEqual(stats['coordinate_scale'], h)
        ref = ref_shapes(xyz, is_data, final, h)
        self.assertEqual(all_voids['GROUP_ID'].tolist(), sorted(ref))
        centers = np.array([ref[g]['center'] for g in all_voids['GROUP_ID']])
        np.testing.assert_allclose(np.column_stack([all_voids['XCART'], all_voids['YCART'], all_voids['ZCART']]),
                                   centers, rtol=1e-12)
        np.testing.assert_allclose(all_voids['R_EFF'], [ref[g]['r_eff'] for g in all_voids['GROUP_ID']], rtol=1e-9)
        np.testing.assert_array_equal(clean['GROUP_ID'], all_voids['GROUP_ID'][~all_voids['BORDER']])
        self.assertEqual(stats['n_border'] + stats['n_clean'], stats['n_groups'])
        self.assertGreater(stats['n_clean'], 20)
        self.assertGreater(stats['n_border'], 0)


class TestCutskyVoidsPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        rng = np.random.default_rng(31)
        centers = np.column_stack([rng.uniform(150, 200, 60), rng.uniform(0, 30, 60), rng.uniform(0.05, 0.3, 60)])
        ra, dec, z = clustered_sky(rng, 9000, centers)
        write_bintable(root / 'real.fits', {'RA': ra, 'DEC': dec, 'Z': z, 'IN_Y5': np.ones(len(ra), dtype=np.int32)})
        for seed, n in ((100, 9000), (200, 9000)):
            write_bintable(root / f'random_S{seed}_1X.fits',
                           {'RA': rng.uniform(150, 200, n), 'DEC': rng.uniform(0, 30, n),
                            'Z': rng.uniform(0.05, 0.3, n), 'IN_Y5': np.ones(n, dtype=np.int32)})
        cls.cfg = dict(real_file=root / 'real.fits', random_glob=str(root / 'random_S*_1X.fits'),
                       output_dir=root / 'out', threads=2, log_interval=3600)
        # ~8000 randoms over ~1300 deg^2: coarse pixels so that most hold >= 3 randoms
        cls.void_cfg = dict(cls.cfg, healpix_nside=32)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_iterations_consensus_and_resume(self):
        # classification of iteration 0 first: the voids reuse its points file
        run_cutsky(CutskyConfig(iteration=0, aggregate=False, **self.cfg))
        for it in (0, 1, 2):
            summary = run_cutsky_voids(VoidCutskyConfig(iteration=it, consensus=it == 2, **self.void_cfg))
            self.assertEqual(summary['status'], 'complete')
        cfg = VoidCutskyConfig(iteration=0, **self.void_cfg)
        paths, run = VoidPaths(cfg.void_dir), RunPaths(cfg.run_dir)
        self.assertEqual(paths.root, cfg.run_dir / 'voids')

        points = np.load(run.points(0))
        cls = np.load(run.classification(0))
        cat = np.load(paths.catalog(0))
        clean = np.load(paths.clean(0))
        mem = np.load(paths.membership(0))
        self.assertEqual((cat.dtype, clean.dtype, mem.dtype), (CATALOG_DTYPE, CLEAN_DTYPE, MEMBERSHIP_DTYPE))
        # membership rows align with the classification run (same catalog and triangulation)
        np.testing.assert_array_equal(mem['TARGETID'], points['TARGETID'])
        np.testing.assert_array_equal(mem['TARGETID'], cls['TARGETID'])
        np.testing.assert_array_equal(mem['R'], cls['R'])
        np.testing.assert_array_equal(mem['SELECTED'], cls['R'] <= -0.25)
        # catalogs and membership agree
        member = mem['GROUP_ID'] >= 0
        counts = np.bincount(mem['GROUP_ID'][member])
        np.testing.assert_array_equal(cat['N_MEMBERS'], counts[cat['GROUP_ID']])
        np.testing.assert_array_equal(cat['N_DATA'], np.bincount(mem['GROUP_ID'][member & mem['ISDATA']],
                                                                 minlength=len(counts))[cat['GROUP_ID']])
        np.testing.assert_array_equal(cat['VOID_ID'], cat['GROUP_ID'].astype(np.int64))
        np.testing.assert_array_equal(mem['VOID_ID'][member], mem['GROUP_ID'][member])
        np.testing.assert_array_equal(clean['VOID_ID'], cat['VOID_ID'][~cat['BORDER']])
        self.assertTrue(np.all(mem['PRUNED'] == ((mem['GROUP_ID_PREMASK'] >= 0) & ~member)))
        self.assertTrue(np.all(cat['BORDER'] == (cat['BORDER_FLAGS'] != 0)))
        self.assertGreater(len(clean), 10)
        self.assertTrue(np.all(cat['N_RANDOM'] > 0) or np.any(np.isnan(cat['R_EFF'])))
        # catalog coordinates in mpc/h: inside the shell of the randoms times h
        h = _core.FlatLambdaCDM.planck18().H0 / 100
        radius = np.sqrt(points['XCART'].astype(float) ** 2 + points['YCART'] ** 2 + points['ZCART'] ** 2)
        center_r = np.sqrt(cat['XCART'] ** 2 + cat['YCART'] ** 2 + cat['ZCART'] ** 2)
        ok = np.isfinite(center_r)
        self.assertTrue(np.all(center_r[ok] <= radius.max() * h * 1.001))
        self.assertTrue(np.all(center_r[ok] >= radius.min() * h * 0.999))
        # iteration 1 has its own VOID_ID range
        cat1 = np.load(paths.catalog(1))
        self.assertTrue(np.all(cat1['VOID_ID'] // VOID_ID_FACTOR == 1))

        # consensus over the three iterations
        cons = np.load(paths.consensus(3))
        self.assertEqual(cons.dtype, CONSENSUS_DTYPE)
        self.assertTrue(np.all(cons['FRAC_V'] > 0.5))
        self.assertTrue(set(np.unique(cons['SRC_ITER'])) <= {0, 1, 2})
        self.assertEqual(len(np.unique(cons['VOID_ID'])), len(cons))
        self.assertEqual(voids_main(['consensus', str(cfg.run_dir), '--iterations', '0-2', '--keep-all',
                                     '--threads', '2']), 0)
        self.assertGreaterEqual(len(np.load(paths.consensus(3, keep_all=True))), len(cons))

        # resume: complete iterations are skipped; other parameters need --overwrite
        self.assertEqual(run_cutsky_voids(cfg)['status'], 'skipped')
        with self.assertRaises(FileExistsError):
            run_cutsky_voids(VoidCutskyConfig(iteration=0, r_threshold=-0.3, **self.void_cfg))
        other = VoidCutskyConfig(iteration=0, r_threshold=-0.3, voids_subdir='voids_r030', membership=False,
                                 **self.void_cfg)
        self.assertEqual(run_cutsky_voids(other)['status'], 'complete')
        self.assertFalse(VoidPaths(other.void_dir).membership(0).exists())

    def test_without_mask(self):
        cfg = VoidCutskyConfig(iteration=4, mask=False, voids_subdir='voids_nomask', save_points=False,
                               **self.void_cfg)
        summary = run_cutsky_voids(cfg)
        self.assertNotIn('mask', summary['stats'])
        cat = np.load(VoidPaths(cfg.void_dir).catalog(4))
        self.assertFalse(cat['BORDER'].any())
        self.assertFalse(RunPaths(cfg.run_dir).points(4).exists())


class TestCubicVoidsPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        rng = np.random.default_rng(5)
        centers = rng.uniform(-50, 50, (30, 3))
        xyz = (centers[rng.integers(0, 30, 6000)] + rng.normal(0, 3, (6000, 3)) + 50) % 100 - 50
        cls.box = root / 'box.fits'
        write_bintable(cls.box, {'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2]})
        cls.cfg = dict(box_file=cls.box, box_size=100.0, output_dir=root / 'out', threads=2, log_interval=3600)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_periodic_box_and_consensus(self):
        for it in (0, 1, 2):
            summary = run_cubic_voids(VoidCubicConfig(iteration=it, **self.cfg))
            self.assertTrue(summary['periodic'])
            self.assertEqual(summary['stats']['unverified'], 0)
            self.assertEqual(summary['stats']['coordinate_scale'], 1.0)
        paths = VoidPaths(VoidCubicConfig(**self.cfg).void_dir)
        cat = np.load(paths.catalog(0))
        self.assertFalse(cat['BORDER'].any())  # periodic: no border
        ok = np.isfinite(cat['XCART'])
        for name in ('XCART', 'YCART', 'ZCART'):
            self.assertTrue(np.all((cat[name][ok] >= -50) & (cat[name][ok] < 50)))
        # the empty background between the blobs percolates through this small box
        self.assertIn('n_wrap_ambiguous', summary['stats'])
        self.assertEqual(voids_main(['consensus', str(paths.root.parent), '--threads', '2']), 0)
        cons = np.load(paths.consensus(3))
        self.assertGreater(len(cons), 0)
        self.assertTrue(np.all((cons['X'] >= -50) & (cons['X'] < 50)))

    def test_nonperiodic_subbox_border(self):
        cfg = VoidCubicConfig(iteration=0, subbox=70.0, membership=False, **self.cfg)
        summary = run_cubic_voids(cfg)
        self.assertFalse(summary['periodic'])
        border = summary['stats']['box_border']
        self.assertGreater(border['n_groups_affected'], 0)
        n_random = summary['n_random']
        self.assertAlmostEqual(border['buffer'], 70.0 / np.cbrt(n_random))
        cat = np.load(VoidPaths(cfg.void_dir).catalog(0))
        self.assertTrue(np.all(cat['BORDER_FLAGS'][cat['BORDER']] == _core.BORDER_BOX_FACE))

    def test_cli(self):
        out = Path(self.tmp.name) / 'cli'
        code = voids_main(['cubic', '--box-file', str(self.box), '--box-size', '100', '--output-dir', str(out),
                           '--threads', '2', '--iteration', '3', '--min-members', '6', '--no-membership'])
        self.assertEqual(code, 0)
        paths = VoidPaths(out / 'cubic' / 'box' / 'voids')
        self.assertTrue(paths.catalog(3).exists())
        self.assertGreaterEqual(np.load(paths.catalog(3))['N_MEMBERS'].min(), 6)


if __name__ == '__main__':
    unittest.main()

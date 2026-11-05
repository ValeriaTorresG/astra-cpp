"""tests of the c++ void finder against a numpy/scipy port of astra-desi
group_finder (branch void-finder): astra.py (watershed), watershed.py (selection
mask and seed-component pruning), shapes.py, and consensus.py.

    python -m unittest discover -s tests -v
"""
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from astra_cpp import _core  # noqa: E402

THREADS = 4


# ---------------------------------------------------------------- synthetic catalogs
def sector(rng, n, rmin, rmax, ra_lim, dec_lim):
    """n points uniform in volume in a spherical shell sector: (r, ra, dec)."""
    r = rng.uniform(rmin ** 3, rmax ** 3, n) ** (1 / 3)
    ra = rng.uniform(*ra_lim, n)
    dec = np.degrees(np.arcsin(rng.uniform(*np.sin(np.radians(dec_lim)), n)))
    return r, ra, dec


def to_xyz(r, ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.column_stack([r * np.cos(dec) * np.cos(ra), r * np.cos(dec) * np.sin(ra), r * np.sin(dec)])


def to_sky(xyz):
    r = np.linalg.norm(xyz, axis=1)
    return r, np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360, np.degrees(np.arcsin(xyz[:, 2] / r))


def cutsky_sample(seed, n=20000):
    """clustered data + as many uniform randoms in a sector with an angular hole
    and a radial gap (both outside the footprint), to exercise the selection mask."""
    rng = np.random.default_rng(seed)
    lims = (300.0, 900.0, (150.0, 200.0), (0.0, 30.0))

    def footprint(r, ra, dec):
        return ~(((ra - 175) ** 2 + (dec - 15) ** 2) < 16) & ~((r > 600) & (r < 612))

    centers = to_xyz(*sector(rng, 300, *lims))
    blobs = centers[rng.integers(0, len(centers), 3 * n)] + rng.normal(0, 8.0, (3 * n, 3))
    pts = np.concatenate([blobs, to_xyz(*sector(rng, 3 * n, *lims))])
    r, ra, dec = to_sky(pts)
    inside = ((r > 300) & (r < 900) & (ra > 150) & (ra < 200) & (dec > 0) & (dec < 30) & footprint(r, ra, dec))
    data = pts[inside][rng.permutation(int(inside.sum()))[:n]]
    rr, rra, rdec = sector(rng, 3 * n, *lims)
    keep = footprint(rr, rra, rdec)
    rr, rra, rdec = rr[keep][:n], rra[keep][:n], rdec[keep][:n]
    xyz = np.concatenate([data, to_xyz(rr, rra, rdec)])
    _, dra, ddec = to_sky(data)
    is_data = np.zeros(len(xyz), dtype=bool)
    is_data[:len(data)] = True
    return xyz, is_data, np.concatenate([dra, rra]), np.concatenate([ddec, rdec])


def box_sample(seed, n, L=100.0):
    rng = np.random.default_rng(seed)
    centers = rng.random((40, 3)) * L
    data = (centers[rng.integers(0, 40, n)] + rng.normal(0, 3.0, (n, 3))) % L
    xyz = np.concatenate([data, rng.random((n, 3)) * L])
    is_data = np.zeros(2 * n, dtype=bool)
    is_data[:n] = True
    return xyz, is_data


# ---------------------------------------------------------------- reference port
def ref_graph(points):
    from scipy.spatial import Delaunay
    indptr, indices = Delaunay(points).vertex_neighbor_vertices
    return indptr, indices


def periodic_graph(points, lo, hi):
    """exact periodic delaunay graph (csr): scipy over the 27 copies of the box."""
    from scipy.spatial import Delaunay
    n = len(points)
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3))) * (np.asarray(hi) - np.asarray(lo))
    indptr, indices = Delaunay(np.concatenate([points + s for s in shifts])).vertex_neighbor_vertices
    center = int(np.flatnonzero(~shifts.any(axis=1))[0])
    rows = []
    for i in range(n):
        v = center * n + i
        nbrs = np.unique(indices[indptr[v]:indptr[v + 1]] % n)
        rows.append(nbrs[nbrs != i])
    return np.concatenate([[0], np.cumsum([len(x) for x in rows])]), np.concatenate(rows)


def ref_r(indptr, indices, is_data):
    """astra.compute_density_contrast."""
    deg = np.diff(indptr)
    nd = np.add.reduceat(is_data[indices].astype(np.int64), indptr[:-1])
    return (nd - (deg - nd)) / deg.astype(np.float64), nd.astype(np.int32), (deg - nd).astype(np.int32)


def ref_group(indptr, indices, r, threshold=-0.25, min_members=4):
    """astra.group_void_points: (group ids after min_members, processing order)."""
    sel = np.flatnonzero(r <= threshold)
    order = sel[np.argsort(r[sel], kind='stable')]
    gid = np.full(len(r), -1, dtype=np.int64)
    nxt = 0
    for p in order:
        nb = gid[indices[indptr[p]:indptr[p + 1]]]
        nb = nb[nb >= 0]
        if len(nb):
            gid[p] = nb.min()
        else:
            gid[p] = nxt
            nxt += 1
    assigned = np.flatnonzero(gid >= 0)
    sizes = np.bincount(gid[assigned], minlength=nxt)
    gid[assigned[sizes[gid[assigned]] < min_members]] = -1
    return gid, order, nxt


def ref_mask(ra, dec, xyz, nside, min_pp, width, min_prb):
    """watershed.build_random_healpix_mask from preloaded randoms."""
    counts = np.bincount(_core.radec2pix_ring(nside, ra, dec), minlength=12 * nside ** 2)
    idx = np.floor(np.linalg.norm(xyz, axis=1) / width).astype(np.int64)
    first, last = idx.min(), idx.max()
    rcounts = np.bincount(idx - first, minlength=last - first + 1)
    edges = np.arange(first, last + 2, dtype=np.float64) * width
    return {'valid_pix': counts >= min_pp, 'edges': edges, 'valid_bins': rcounts >= min_prb,
            'counts': counts, 'rcounts': rcounts, 'nside': nside}


def ref_apply_mask(points, is_data, ra, dec, indptr, indices, gid_before, order, mask, min_members=4, step=None):
    """watershed.apply_random_healpix_edge_mask + _prune_to_seed_connected_components."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(points)
    nside = mask['nside']
    step = step or 0.5 * np.sqrt(4 * np.pi / (12 * nside ** 2))
    radius = np.linalg.norm(points, axis=1)
    valid_pixels = np.all(np.isfinite(points), axis=1) & (radius > 0)
    pixels = np.full(n, -1, dtype=np.int64)
    unit = points[valid_pixels] / radius[valid_pixels, None]
    pixels[valid_pixels] = _core.vec2pix_ring(nside, unit[:, 0], unit[:, 1], unit[:, 2])
    rnd = ~is_data
    pixels[rnd] = _core.radec2pix_ring(nside, ra[rnd], dec[rnd])
    valid_pixels[rnd] = np.isfinite(ra[rnd]) & (np.abs(dec[rnd]) <= 90)
    retained = gid_before >= 0
    inv_ang = retained & ~valid_pixels
    use = np.flatnonzero(retained & valid_pixels)
    inv_ang[use] = ~mask['valid_pix'][pixels[use]]
    edges_r = mask['edges']
    valid_r = (radius > 0) & (radius >= edges_r[0]) & (radius <= edges_r[-1])
    bins = np.clip(np.searchsorted(edges_r, radius, side='right') - 1, 0, len(edges_r) - 2)
    inv_rad = retained & ~valid_r
    use = np.flatnonzero(retained & valid_r)
    inv_rad[use] = ~mask['valid_bins'][bins[use]]

    owner = np.repeat(np.arange(n), np.diff(indptr))
    upper = indices > owner
    e = np.column_stack([owner[upper], indices[upper]])
    internal = (gid_before[e[:, 0]] >= 0) & (gid_before[e[:, 0]] == gid_before[e[:, 1]])
    ie = e[internal]
    # angular crossings (great-circle samples)
    p0, p1 = points[ie[:, 0]], points[ie[:, 1]]
    r0, r1 = np.linalg.norm(p0, axis=1), np.linalg.norm(p1, axis=1)
    keep = valid_pixels[ie[:, 0]] & valid_pixels[ie[:, 1]] & (r0 > 0) & (r1 > 0)
    keep &= mask['valid_pix'][pixels[ie[:, 0]]] & mask['valid_pix'][pixels[ie[:, 1]]]
    u0, u1 = p0 / r0[:, None], p1 / r1[:, None]
    angles = np.arccos(np.clip(np.einsum('ij,ij->i', u0, u1), -1, 1))
    keep &= angles < np.pi - 1e-10
    nseg = np.maximum(1, np.ceil(angles / step).astype(np.int64))
    for s in range(1, int(nseg.max()) if len(nseg) else 1):
        act = np.flatnonzero(keep & (nseg > s))
        if not len(act):
            continue
        f = s / nseg[act]
        a = angles[act]
        sa = np.sin(a)
        w0, w1 = np.sin((1 - f) * a) / sa, np.sin(f * a) / sa
        v = w0[:, None] * u0[act] + w1[:, None] * u1[act]
        v /= np.linalg.norm(v, axis=1)[:, None]
        keep[act] &= mask['valid_pix'][_core.vec2pix_ring(nside, v[:, 0], v[:, 1], v[:, 2])]
    ang_cross = ~keep
    # radial crossings
    d = p1 - p0
    dsq = np.einsum('ij,ij->i', d, d)
    t = np.clip(np.where(dsq > 0, -np.einsum('ij,ij->i', p0, d) / np.where(dsq > 0, dsq, 1), 0), 0, 1)
    rmin = np.linalg.norm(p0 + t[:, None] * d, axis=1)
    rmax = np.maximum(r0, r1)
    support = (rmin > 0) & (rmin >= edges_r[0]) & (rmax <= edges_r[-1])
    prefix = np.concatenate([[0], np.cumsum(~mask['valid_bins'])])
    fb = np.clip(np.searchsorted(edges_r, rmin, side='right') - 1, 0, len(edges_r) - 2)
    lb = np.clip(np.searchsorted(edges_r, rmax, side='right') - 1, 0, len(edges_r) - 2)
    rad_cross = ~(support & (prefix[lb + 1] - prefix[fb] == 0))

    g = gid_before
    border = {'angular_member': set(np.unique(g[inv_ang]).tolist()),
              'radial_member': set(np.unique(g[inv_rad]).tolist()),
              'angular_edge': set(np.unique(g[ie[ang_cross, 0]]).tolist()),
              'radial_edge': set(np.unique(g[ie[rad_cross, 0]]).tolist())}
    # seed-connected component of each group
    invalid = inv_ang | inv_rad
    valid_member = retained & ~invalid
    good = ~ang_cross & ~rad_cross & valid_member[ie[:, 0]] & valid_member[ie[:, 1]]
    graph = coo_matrix((np.ones(int(good.sum())), (ie[good, 0], ie[good, 1])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    seed = {}
    for p in order:
        if g[p] >= 0 and g[p] not in seed:
            seed[g[p]] = p
    final = np.full(n, -1, dtype=np.int64)
    for grp, s in seed.items():
        if invalid[s]:
            continue
        members = np.flatnonzero((g == grp) & valid_member & (comp == comp[s]))
        if len(members) >= min_members:
            final[members] = grp
    stats = {'n_internal': int(internal.sum()), 'n_ang': int(ang_cross.sum()), 'n_rad': int(rad_cross.sum())}
    return final, border, stats


def ref_shapes(points, is_data, gid, scale=1.0):
    """shapes.compute_void_shapes: {group: (center, eigval, eigvec, r_eff, ellip, n_data, n_random)}."""
    out = {}
    for grp in np.unique(gid[gid >= 0]):
        members = np.flatnonzero(gid == grp)
        rnd = members[~is_data[members]]
        res = {'n_data': int(is_data[members].sum()), 'n_random': len(rnd), 'center': np.full(3, np.nan),
               'eigval': np.full(3, np.nan), 'eigvec': np.full((3, 3), np.nan), 'r_eff': np.nan, 'ellip': np.nan}
        if len(rnd):
            pos = points[rnd] * scale
            res['center'] = pos.mean(axis=0)
            c = pos - res['center']
            shape = c.T @ c / len(pos)
            w, v = np.linalg.eigh(0.5 * (shape + shape.T))
            w, v = w[::-1], v[:, ::-1].T.copy()
            eps = np.finfo(np.float64).eps
            ok = not np.any(w < -64 * eps * max(np.abs(w).max(), np.finfo(np.float64).tiny))
            w = np.clip(w, 0, None)
            ok = ok and w[0] > 0 and w[2] > 3 * eps * w[0]
            if ok:
                for a in range(3):
                    if v[a, np.argmax(np.abs(v[a]))] < 0:
                        v[a] *= -1
                res.update(eigval=w, eigvec=v, r_eff=np.sqrt(5) * np.exp(np.log(w).sum() / 6),
                           ellip=1 - np.clip((w[2] + w[1]) / (w[1] + w[0]), 0, 1) ** 0.25)
        out[int(grp)] = res
    return out


def ref_consensus(centers, radii, origins, n_runs, vol_frac=0.5):
    """consensus.build_consensus_catalog steps 2-5: (pooled rows, support, n_absorbed)."""
    from scipy.spatial import cKDTree

    def sweep(c, r, pick):
        tree = cKDTree(c)
        absorbed = np.zeros(len(r), dtype=bool)
        out = []
        for seed in np.argsort(-r, kind='stable'):
            if absorbed[seed]:
                continue
            near = np.asarray(tree.query_ball_point(c[seed], r[seed]), dtype=np.int64)
            frac = _core.contained_volume_fraction(np.linalg.norm(c[near] - c[seed], axis=1), r[seed], r[near])
            group = near[frac > vol_frac]
            if not np.any(group == seed):
                group = np.append(group, seed)
            out.append(pick(group))
            absorbed[group] = True
        return out

    def medoid(group):
        best = {}
        for j in sorted(group.tolist()):
            o = origins[j]
            if o not in best or radii[j] > radii[best[o]]:
                best[o] = j
        votes = sorted(best.values(), key=lambda j: (radii[j], j))
        return votes[(len(votes) - 1) // 2], len(votes)

    groups = sweep(centers, radii, medoid)
    med = np.array([g[0] for g in groups])
    support = np.array([g[1] for g in groups])
    # largest representative of each group; ties (the same pooled void can represent
    # two groups) go to the lowest representative index
    kept = sweep(centers[med], radii[med], lambda g: (np.sort(g)[np.argmax(radii[med][np.sort(g)])], len(g) - 1))
    return med[[k for k, _ in kept]], support[[k for k, _ in kept]], np.array([a for _, a in kept])


# ---------------------------------------------------------------- tests
def run_finder(points, is_data, threshold=-0.25, min_members=4, threads=THREADS, **tri_kw):
    tri = _core.Delaunay3(points, threads=1, **tri_kw)
    nd, nr = tri.neighbor_counts(is_data, threads=threads)
    vf = _core.VoidFinder(nd, nr, threshold, threads=threads)
    vf.build_adjacency(tri, threads=threads)
    vf.grow(min_members)
    return vf, nd, nr


def per_point(vf, by_rank, n):
    out = np.full(n, -1, dtype=np.int64)
    out[vf.order] = by_rank
    return out


class TestHealpix(unittest.TestCase):
    @staticmethod
    def pix2ang_ring(nside, p):
        """pixel centers in the ring scheme (healpix pix2ang_ring)."""
        p = np.asarray(p, dtype=np.int64)
        npix, ncap = 12 * nside ** 2, 2 * nside * (nside - 1)
        z, phi = np.empty(len(p)), np.empty(len(p))
        north, south = p < ncap, p >= npix - ncap
        eq = ~north & ~south
        i = np.floor((1 + np.sqrt(1 + 2 * p[north])) / 2).astype(np.int64)
        j = p[north] + 1 - 2 * i * (i - 1)
        z[north], phi[north] = 1 - i ** 2 / (3 * nside ** 2), np.pi / (2 * i) * (j - 0.5)
        pp = p[eq] - ncap
        i, j = pp // (4 * nside) + nside, pp % (4 * nside) + 1
        fodd = np.where((i + nside) % 2 == 1, 1.0, 0.5)  # unshifted rings start at phi = 0
        z[eq], phi[eq] = 4 / 3 - 2 * i / (3 * nside), np.pi / (2 * nside) * (j - fodd)
        pp = npix - p[south]
        i = np.floor((1 + np.sqrt(2 * pp - 1)) / 2).astype(np.int64)
        j = 4 * i + 1 - (pp - 2 * i * (i - 1))
        z[south], phi[south] = -1 + i ** 2 / (3 * nside ** 2), np.pi / (2 * i) * (j - 0.5)
        return np.arccos(z), phi

    def test_pixel_centers_round_trip(self):
        for nside in (1, 2, 4, 16, 64, 128):
            with self.subTest(nside=nside):
                p = np.arange(12 * nside ** 2)
                theta, phi = self.pix2ang_ring(nside, p)
                np.testing.assert_array_equal(_core.ang2pix_ring(nside, theta, phi), p)
                v = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
                np.testing.assert_array_equal(_core.vec2pix_ring(nside, *(3.7 * v).T), p)

    def test_random_points_equal_area(self):
        rng = np.random.default_rng(3)
        nside, n = 16, 2_000_000
        v = rng.normal(size=(n, 3))
        pix = _core.vec2pix_ring(nside, *v.T)
        counts = np.bincount(pix, minlength=12 * nside ** 2)
        expected = n / len(counts)
        self.assertLess(np.abs(counts - expected).max(), 6 * np.sqrt(expected))
        # every point lies close to its pixel center
        theta, phi = self.pix2ang_ring(nside, pix[:20000])
        c = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        u = v[:20000] / np.linalg.norm(v[:20000], axis=1)[:, None]
        self.assertLess(np.degrees(np.arccos(np.clip((u * c).sum(1), -1, 1))).max(), 2.5 * 58.6 / nside)
        # radec2pix agrees with vec2pix of the same direction
        ra = np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 360
        dec = np.degrees(np.arcsin(u[:, 2]))
        agree = _core.radec2pix_ring(nside, ra, dec) == pix[:20000]
        self.assertGreater(agree.mean(), 0.9999)


class TestWatershed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xyz, cls.is_data, cls.ra, cls.dec = cutsky_sample(5, 15000)
        cls.indptr, cls.indices = ref_graph(cls.xyz)
        cls.r, cls.nd, cls.nr = ref_r(cls.indptr, cls.indices, cls.is_data)

    def test_order_and_rank(self):
        for thr in (-0.25, -0.5, 0.0):
            with self.subTest(threshold=thr):
                vf = _core.VoidFinder(self.nd, self.nr, thr, threads=THREADS)
                sel = np.flatnonzero(self.r <= thr)
                np.testing.assert_array_equal(vf.order, sel[np.argsort(self.r[sel], kind='stable')])
                rank = vf.rank
                np.testing.assert_array_equal(rank[vf.order], np.arange(vf.n_selected))
                self.assertTrue(np.all(rank[self.r > thr] == -1))
                self.assertEqual(vf.count_selected(0, int(self.is_data.sum())), int((self.r[self.is_data] <= thr).sum()))

    def test_lower_adjacency_is_induced_graph(self):
        vf, _, _ = run_finder(self.xyz, self.is_data)
        offsets, nbrs = vf.adjacency()
        order, rank = vf.order, vf.rank
        owner = np.repeat(np.arange(len(self.xyz)), np.diff(self.indptr))
        both = (rank[owner] >= 0) & (rank[self.indices] >= 0)
        a, b = rank[owner[both]], rank[self.indices[both]]
        expected = set(zip(np.maximum(a, b).tolist(), np.minimum(a, b).tolist()))
        rows = np.repeat(np.arange(len(order)), np.diff(offsets))
        got = list(zip(rows.tolist(), nbrs.tolist()))
        self.assertEqual(len(got), len(set(got)))
        self.assertEqual(set(got), expected)
        self.assertTrue(np.all(nbrs < rows))

    def test_groups_match_reference(self):
        for thr, mm in ((-0.25, 4), (-0.4, 1), (0.1, 10)):
            with self.subTest(threshold=thr, min_members=mm):
                vf, _, _ = run_finder(self.xyz, self.is_data, thr, mm)
                gid, _, n_seeds = ref_group(self.indptr, self.indices, self.r, thr, mm)
                self.assertEqual(vf.n_seeds, n_seeds)
                np.testing.assert_array_equal(per_point(vf, vf.premask_by_rank, len(gid)), gid)
                premask, group, flags = vf.membership()
                np.testing.assert_array_equal(premask, gid)
                np.testing.assert_array_equal(group, gid)  # no mask
                np.testing.assert_array_equal((flags & _core.MEMBER_SELECTED) != 0, self.r <= thr)

    def test_thread_count_does_not_change_results(self):
        a, _, _ = run_finder(self.xyz, self.is_data, threads=1)
        b, _, _ = run_finder(self.xyz, self.is_data, threads=7)
        np.testing.assert_array_equal(a.order, b.order)
        np.testing.assert_array_equal(a.premask_by_rank, b.premask_by_rank)


class TestSelectionMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xyz, cls.is_data, cls.ra, cls.dec = cutsky_sample(9, 20000)
        cls.indptr, cls.indices = ref_graph(cls.xyz)
        cls.r, _, _ = ref_r(cls.indptr, cls.indices, cls.is_data)
        cls.gid, cls.order, _ = ref_group(cls.indptr, cls.indices, cls.r)
        cls.rnd = ~cls.is_data

    def mask_args(self, nside):
        return (self.ra[self.rnd], self.dec[self.rnd], np.ascontiguousarray(self.xyz[self.rnd]), nside, 3, 5.0, 3)

    def test_mask_counts(self):
        mask = _core.build_selection_mask(*self.mask_args(64), threads=THREADS)
        ref = ref_mask(*self.mask_args(64))
        np.testing.assert_array_equal(mask.pixel_counts, ref['counts'])
        np.testing.assert_array_equal(mask.radial_counts, ref['rcounts'])
        np.testing.assert_array_equal(mask.radial_edges, ref['edges'])
        self.assertEqual(mask.n_valid_pixels, int(ref['valid_pix'].sum()))
        self.assertGreater(len(ref['rcounts']) - mask.n_valid_radial_bins, 0)  # the radial gap

    def test_pruning_matches_reference(self):
        for nside in (64, 128):
            with self.subTest(nside=nside):
                vf, _, _ = run_finder(self.xyz, self.is_data)
                mask = _core.build_selection_mask(*self.mask_args(nside), threads=THREADS)
                step = 0.5 * np.sqrt(4 * np.pi / (12 * nside ** 2))
                st = vf.apply_selection_mask(mask, self.xyz, self.is_data, self.ra, self.dec, step, 4, threads=THREADS)
                final, border, rst = ref_apply_mask(self.xyz, self.is_data, self.ra, self.dec, self.indptr,
                                                    self.indices, self.gid, self.order, ref_mask(*self.mask_args(nside)))
                self.assertEqual(st['n_internal_edges'], rst['n_internal'])
                self.assertEqual(st['n_angular_edge_crossings'], rst['n_ang'])
                self.assertEqual(st['n_radial_edge_crossings'], rst['n_rad'])
                self.assertGreater(rst['n_ang'], 0)
                self.assertGreater(rst['n_rad'], 0)
                np.testing.assert_array_equal(per_point(vf, vf.group_by_rank, len(final)), final)
                flags = vf.border
                for key, bit in (('angular_member', _core.BORDER_ANGULAR_MEMBER),
                                 ('radial_member', _core.BORDER_RADIAL_MEMBER),
                                 ('angular_edge', _core.BORDER_ANGULAR_EDGE),
                                 ('radial_edge', _core.BORDER_RADIAL_EDGE)):
                    self.assertEqual(set(np.flatnonzero(flags & bit).tolist()), border[key], key)
                pruned = (self.gid >= 0) & (final < 0)
                self.assertEqual(st['n_pruned_members'], int(pruned.sum()))
                premask, group, mflags = vf.membership()
                np.testing.assert_array_equal((mflags & _core.MEMBER_PRUNED) != 0, pruned)
                any_border = set().union(*border.values())
                np.testing.assert_array_equal((mflags & _core.MEMBER_BORDER) != 0,
                                              np.isin(self.gid, list(any_border)) & (self.gid >= 0))

    def test_shapes_match_reference(self):
        vf, _, _ = run_finder(self.xyz, self.is_data)
        mask = _core.build_selection_mask(*self.mask_args(64), threads=THREADS)
        vf.apply_selection_mask(mask, self.xyz, self.is_data, self.ra, self.dec, 0.5 * np.sqrt(4 * np.pi / 49152), 4)
        h = 0.6766
        s = vf.shapes(self.xyz, self.is_data, scale=h, threads=THREADS)
        final = per_point(vf, vf.group_by_rank, len(self.xyz))
        ref = ref_shapes(self.xyz, self.is_data, final, h)
        self.assertEqual(s['group_id'].tolist(), sorted(ref))
        self.assertGreater(np.isfinite(s['r_eff']).sum(), 50)
        for k, g in enumerate(s['group_id']):
            e = ref[int(g)]
            self.assertEqual((s['n_data'][k], s['n_random'][k]), (e['n_data'], e['n_random']))
            np.testing.assert_allclose(s['center'][k], e['center'], rtol=1e-12, atol=1e-9)
            if np.isnan(e['r_eff']):
                self.assertTrue(np.isnan(s['r_eff'][k]))
                continue
            scale = e['eigval'][0]
            np.testing.assert_allclose(s['eigval'][k], e['eigval'], rtol=0, atol=1e-10 * scale)
            np.testing.assert_allclose(s['r_eff'][k], e['r_eff'], rtol=1e-9)
            np.testing.assert_allclose(s['ellip'][k], e['ellip'], rtol=1e-7, atol=1e-12)
            gap = np.min(np.diff(e['eigval'][::-1])) / scale
            if gap > 1e-6:  # well separated axes: eigenvectors are unique up to the sign convention
                np.testing.assert_allclose(s['eigvec'][k], e['eigvec'], atol=1e-6)


class TestPeriodic(unittest.TestCase):
    L = 100.0

    @classmethod
    def setUpClass(cls):
        cls.xyz, cls.is_data = box_sample(2, 1500, cls.L)
        cls.box = dict(box_lo=[0.0] * 3, box_hi=[cls.L] * 3, ghost_margin=25.0)

    def test_groups_match_exact_periodic_graph(self):
        vf, _, _ = run_finder(self.xyz, self.is_data, **self.box)
        indptr, indices = periodic_graph(self.xyz, [0] * 3, [self.L] * 3)
        r, _, _ = ref_r(indptr, indices, self.is_data)
        gid, _, n_seeds = ref_group(indptr, indices, r)
        self.assertEqual(vf.n_seeds, n_seeds)
        np.testing.assert_array_equal(per_point(vf, vf.premask_by_rank, len(gid)), gid)

    def test_translation_invariance(self):
        """a periodic shift moves every void center by the same shift."""
        shift = np.array([37.0, 61.0, 83.0])
        moved = (self.xyz + shift) % self.L
        box = {'box_lo': self.box['box_lo'], 'box_hi': self.box['box_hi']}
        a, _, _ = run_finder(self.xyz, self.is_data, **self.box)
        b, _, _ = run_finder(moved, self.is_data, **self.box)
        np.testing.assert_array_equal(a.premask_by_rank, b.premask_by_rank)
        sa = a.shapes(self.xyz, self.is_data, **box)
        sb = b.shapes(moved, self.is_data, **box)
        np.testing.assert_array_equal(sa['group_id'], sb['group_id'])
        ok = np.isfinite(sa['r_eff'])
        self.assertGreater(ok.sum(), 10)
        delta = (sb['center'] - sa['center'] - shift + self.L / 2) % self.L - self.L / 2
        np.testing.assert_allclose(delta, 0, atol=1e-9)
        self.assertTrue(np.all((sb['center'] >= 0) & (sb['center'] < self.L)))
        np.testing.assert_allclose(sa['eigval'][ok], sb['eigval'][ok], rtol=1e-8)
        np.testing.assert_allclose(sa['r_eff'][ok], sb['r_eff'][ok], rtol=1e-9)
        # without unwrapping, groups that straddle a face would look much larger
        naive = b.shapes(moved, self.is_data)
        self.assertTrue(np.any(naive['r_eff'][ok] > 1.5 * sb['r_eff'][ok]))

    def test_box_border(self):
        vf, _, _ = run_finder(self.xyz, self.is_data)
        n = vf.flag_box_border(self.xyz, [0.0] * 3, [self.L] * 3, 5.0)
        gid = per_point(vf, vf.premask_by_rank, len(self.xyz))
        near = np.any((self.xyz <= 5.0) | (self.L - self.xyz <= 5.0), axis=1) & (gid >= 0)
        expected = set(np.unique(gid[near]).tolist())
        self.assertEqual(set(np.flatnonzero(vf.border & _core.BORDER_BOX_FACE).tolist()), expected)
        self.assertEqual(n, len(expected))


class TestDuplicates(unittest.TestCase):
    def test_duplicates_join_their_canonical_point(self):
        xyz, is_data = box_sample(4, 3000)
        sel = np.arange(3000, 3020)  # randoms (likely in voids)
        pts = np.concatenate([xyz, xyz[sel]])
        flags = np.concatenate([is_data, np.zeros(len(sel), dtype=bool)])
        vf, nd, nr = run_finder(pts, flags, threshold=0.5)
        dup = np.arange(len(xyz), len(pts))
        np.testing.assert_array_equal(nd[dup], nd[sel])
        self.assertEqual(vf.n_merged_duplicates, len(sel))
        premask, group, mflags = vf.membership()
        np.testing.assert_array_equal(premask[dup], premask[sel])
        np.testing.assert_array_equal(group[dup], group[sel])
        np.testing.assert_array_equal(mflags[dup], mflags[sel])
        self.assertTrue(np.all(mflags[dup] & _core.MEMBER_SELECTED))
        # merged duplicates count as members of their canonical point's group
        s = vf.shapes(pts, flags)
        counts = np.bincount(group[group >= 0])
        np.testing.assert_array_equal(s['n_members'], counts[s['group_id']])


class TestConsensus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(8)
        n_runs, truth = 6, rng.random((300, 3)) * 400
        true_r = rng.uniform(5, 25, len(truth))
        parts = []
        for run in range(n_runs):
            keep = rng.random(len(truth)) < 0.7
            c = truth[keep] + rng.normal(0, 2.0, (keep.sum(), 3))
            r = true_r[keep] * rng.uniform(0.85, 1.15, keep.sum())
            extra = rng.integers(20, 40)
            c = np.concatenate([c, rng.random((extra, 3)) * 400])
            r = np.concatenate([r, rng.uniform(3, 10, extra)])
            parts.append((c, r, np.full(len(r), run, dtype=np.int32)))
        cls.c = np.concatenate([p[0] for p in parts])
        cls.r = np.concatenate([p[1] for p in parts])
        cls.o = np.concatenate([p[2] for p in parts])
        cls.n_runs = n_runs

    def test_volume_fraction(self):
        f = _core.contained_volume_fraction
        self.assertEqual(f(0.0, 10.0, 5.0), 1.0)
        self.assertAlmostEqual(f(0.0, 5.0, 10.0), 0.125)
        self.assertEqual(f(20.0, 10.0, 5.0), 0.0)
        # half of an equal sphere at d = 0 ... 2r: numeric volume check
        rng = np.random.default_rng(1)
        p = rng.uniform(-1, 1, (400000, 3))
        p = p[np.linalg.norm(p, axis=1) <= 1]
        for d in (0.3, 0.9, 1.5):
            inside = np.linalg.norm(p - [d, 0, 0], axis=1) <= 1.2
            self.assertAlmostEqual(f(d, 1.2, 1.0), inside.mean(), delta=0.005)

    def test_matches_reference(self):
        for vol in (0.5, 0.7):
            with self.subTest(vol_frac=vol):
                res = _core.void_consensus(self.c, self.r, self.o, self.n_runs, vol_frac=vol, batch=37, threads=THREADS)
                rows, support, absorbed = ref_consensus(self.c, self.r, self.o, self.n_runs, vol)
                np.testing.assert_array_equal(res['rows'], rows)
                np.testing.assert_array_equal(res['support'], support)
                np.testing.assert_array_equal(res['n_absorbed'], absorbed)
                self.assertGreater((res['support'] / self.n_runs > 0.5).sum(), 100)

    def test_periodic_consensus_is_translation_invariant(self):
        L = 400.0
        a = _core.void_consensus(self.c % L, self.r, self.o, self.n_runs, box_lo=[0] * 3, box_hi=[L] * 3)
        b = _core.void_consensus((self.c + [150, 250, 330]) % L, self.r, self.o, self.n_runs, box_lo=[0] * 3,
                                 box_hi=[L] * 3)
        np.testing.assert_array_equal(a['rows'], b['rows'])
        np.testing.assert_array_equal(a['support'], b['support'])


if __name__ == '__main__':
    unittest.main()

"""astra void finder on one catalog: delaunay -> r -> watershed -> selection -> shapes.

all heavy stages run in c++ (`_core.VoidFinder`); this module wires them and
assembles the output arrays.
"""
from __future__ import annotations

import logging

import numpy as np

from .. import _core
from ..catalogs import Catalog
from ..monitor import ResourceMonitor
from ..pipeline import neighbor_counts, triangulate
from .config import PEAK_BYTES_PER_POINT, VOID_ID_FACTOR, VoidParams
from .outputs import CATALOG_DTYPE, CLEAN_DTYPE, EIGVAL_FIELDS, EIGVEC_FIELDS, MEMBERSHIP_DTYPE


def planck18_h() -> float:
    return _core.FlatLambdaCDM.planck18().H0 / 100.0


def coordinate_scale(cat: Catalog, params: VoidParams) -> float:
    """catalog coordinates are in mpc/h: positions in mpc (cut-sky) are multiplied by h."""
    if params.h > 0:
        return float(params.h)
    unit = cat.meta.get('distance_unit', 'Mpc')
    return 1.0 if unit == 'Mpc/h' else planck18_h()


def box_bounds(cat: Catalog) -> tuple[list[float], list[float]] | None:
    if 'box_lo' in cat.meta:
        return list(map(float, cat.meta['box_lo'])), list(map(float, cat.meta['box_hi']))
    return None


def void_ids(group_id: np.ndarray, iteration: int) -> np.ndarray:
    if len(group_id) and int(np.max(group_id)) >= VOID_ID_FACTOR:
        raise ValueError('group ids do not fit in VOID_ID')
    return iteration * VOID_ID_FACTOR + group_id.astype(np.int64)


def build_catalogs(shapes: dict, border_flags: np.ndarray, iteration: int) -> tuple[np.ndarray, np.ndarray]:
    """(all, clean) structured catalogs, one row per final group, ordered by group id."""
    gid = shapes['group_id']
    flags = border_flags[gid] if len(gid) else np.zeros(0, dtype=np.uint8)
    out = np.zeros(len(gid), dtype=CATALOG_DTYPE)
    out['VOID_ID'] = void_ids(gid, iteration)
    out['GROUP_ID'] = gid
    for k, name in enumerate(('XCART', 'YCART', 'ZCART')):
        out[name] = shapes['center'][:, k]
    out['R_EFF'] = shapes['r_eff']
    out['ELLIP'] = shapes['ellip']
    for a, name in enumerate(EIGVAL_FIELDS):
        out[name] = shapes['eigval'][:, a]
    for k, name in enumerate(EIGVEC_FIELDS):
        out[name] = shapes['eigvec'][:, k // 3, k % 3]
    out['N_MEMBERS'] = shapes['n_members']
    out['N_DATA'] = shapes['n_data']
    out['N_RANDOM'] = shapes['n_random']
    out['BORDER'] = flags != 0
    out['BORDER_FLAGS'] = flags
    clean = np.zeros(int(np.count_nonzero(~out['BORDER'])), dtype=CLEAN_DTYPE)
    keep = ~out['BORDER']
    for name in CLEAN_DTYPE.names:
        clean[name] = out[name][keep]
    return out, clean


def membership_columns(vf, cat: Catalog, r: np.ndarray, iteration: int, threads: int) -> dict[str, np.ndarray]:
    premask, group, flags = vf.membership(threads=threads)
    vid = np.full(len(group), -1, dtype=np.int64)
    member = group >= 0
    vid[member] = void_ids(group[member], iteration)
    return {'TARGETID': cat.targetid, 'ISDATA': cat.is_data, 'R': r,
            'SELECTED': (flags & _core.MEMBER_SELECTED) != 0, 'GROUP_ID_PREMASK': premask, 'GROUP_ID': group,
            'VOID_ID': vid, 'PRUNED': (flags & _core.MEMBER_PRUNED) != 0,
            'BORDER': (flags & _core.MEMBER_BORDER) != 0}


def find_voids(cat: Catalog, params, iteration: int, log: logging.Logger, mon: ResourceMonitor,
               membership: bool = True) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray] | None, dict]:
    """run the void finder on `cat` (data rows first). `params` is a void config
    (AstraParams + VoidParams). returns (all catalog, clean catalog, membership
    columns or None, stats)."""
    threads = params.threads
    tri, stats, periodic = triangulate(cat, params, log, mon, PEAK_BYTES_PER_POINT)
    ndata, nrand = neighbor_counts(tri, cat, params, periodic, stats, log, mon)

    with mon.stage('orden_r'):
        vf = _core.VoidFinder(ndata, nrand, params.r_threshold, threads=threads)
    log.info('r <= %g: %s of %s points selected (%.4f)', params.r_threshold, vf.n_selected, cat.size,
             vf.n_selected / max(1, cat.size))
    with mon.stage('adyacencia'):
        n_edges = vf.build_adjacency(tri, threads=threads)
    del tri
    log.info('induced delaunay graph of the selected points: %s edges (%.2f per point)',
             n_edges, 2 * n_edges / max(1, vf.n_selected))

    with mon.stage('watershed'):
        vf.grow(params.min_members)
    log.info('watershed: %s seeds, %s groups with >= %s members', vf.n_seeds, vf.n_groups_premask,
             params.min_members)
    stats.update(r_threshold=params.r_threshold, n_selected=int(vf.n_selected),
                 n_selected_data=int(vf.count_selected(0, cat.n_data)),
                 n_induced_edges=int(n_edges), n_seeds=int(vf.n_seeds),
                 n_groups_before_mask=int(vf.n_groups_premask))

    box = box_bounds(cat)
    if 'RA' in cat.columns and params.mask:
        rand = slice(cat.n_data, None)
        ra, dec = cat.columns['RA'], cat.columns['DEC']
        with mon.stage('mascara'):
            mask = _core.build_selection_mask(ra[rand], dec[rand], cat.xyz[rand], nside=params.healpix_nside,
                                              min_randoms_per_pixel=params.min_randoms_per_pixel,
                                              radial_bin_width=params.radial_bin_width,
                                              min_randoms_per_radial_bin=params.min_randoms_per_radial_bin,
                                              threads=threads)
            mstats = vf.apply_selection_mask(mask, cat.xyz, cat.is_data, ra, dec,
                                             angular_sample_step=params.sample_step(),
                                             min_members=params.min_members, threads=threads)
        stats['mask'] = {'nside': int(mask.nside), 'n_random_rows': int(mask.n_rows),
                         'n_nonempty_pixels': int(mask.n_nonempty_pixels), 'n_valid_pixels': int(mask.n_valid_pixels),
                         'n_radial_bins': int(len(mask.radial_counts)),
                         'n_valid_radial_bins': int(mask.n_valid_radial_bins),
                         'radial_min': float(mask.radial_edges[0]), 'radial_max': float(mask.radial_edges[-1]),
                         'angular_sample_step': params.sample_step(), **mstats}
        log.info('selection mask: %s/%s valid pixels (nside=%s), %s/%s valid radial bins; groups affected=%s '
                 'pruned=%s discarded=%s', mask.n_valid_pixels, mask.n_nonempty_pixels, mask.nside,
                 mask.n_valid_radial_bins, len(mask.radial_counts), mstats['n_groups_affected'],
                 mstats['n_groups_pruned'], mstats['n_groups_discarded'])
        del mask
    elif box is not None and not periodic:
        lo, hi = box
        n_random = cat.size - cat.n_data
        buffer = params.boundary_buffer
        if buffer < 0:  # mean random spacing
            buffer = float(np.cbrt(np.prod(np.subtract(hi, lo)) / max(1, n_random)))
        n_border = vf.flag_box_border(cat.xyz, lo, hi, buffer, threads=threads)
        stats['box_border'] = {'buffer': buffer, 'n_groups_affected': int(n_border)}
        log.info('nonperiodic box: %s groups within %.3f of a face are border', n_border, buffer)
    vf.release_adjacency()

    scale = coordinate_scale(cat, params)
    shape_box = {'box_lo': box[0], 'box_hi': box[1]} if (box is not None and periodic) else {}
    with mon.stage('formas'):
        shapes = vf.shapes(cat.xyz, cat.is_data, scale=scale, threads=threads, **shape_box)
    all_voids, clean = build_catalogs(shapes, vf.border, iteration)
    n_ambiguous = int(np.count_nonzero(shapes['wrap_ambiguous']))
    if n_ambiguous:
        log.warning('%s groups span more than half the periodic box along some axis: their unwrapped '
                    'shapes are ambiguous', n_ambiguous)
    stats['n_wrap_ambiguous'] = n_ambiguous
    stats['n_merged_duplicates'] = int(vf.n_merged_duplicates)
    del shapes
    stats.update(coordinate_scale=scale, n_groups=int(len(all_voids)),
                 n_border=int(np.count_nonzero(all_voids['BORDER'])), n_clean=int(len(clean)),
                 n_defined_shapes=int(np.count_nonzero(np.isfinite(all_voids['R_EFF']))),
                 n_member_points=int(all_voids['N_MEMBERS'].sum()))
    if len(clean):
        finite = np.isfinite(clean['R_EFF'])
        if np.any(finite):
            q = np.percentile(clean['R_EFF'][finite], (16, 50, 84))
            stats['clean_r_eff_percentiles'] = {'p16': float(q[0]), 'p50': float(q[1]), 'p84': float(q[2])}
    log.info('voids: %s groups (%s border, %s clean, %s with a defined shape)', stats['n_groups'],
             stats['n_border'], stats['n_clean'], stats['n_defined_shapes'])

    columns = None
    if membership:
        with mon.stage('membresia'):
            r, _ = _core.classify(ndata, nrand, threads=threads)
            columns = membership_columns(vf, cat, r, iteration, threads)
    return all_voids, clean, columns, stats


__all__ = ['find_voids', 'build_catalogs', 'coordinate_scale', 'void_ids', 'planck18_h', 'MEMBERSHIP_DTYPE']

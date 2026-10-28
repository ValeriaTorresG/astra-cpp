import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from cutsky_ra_distance_wedge import ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif'})


LABELS = ('Void', 'Sheet', 'Filament', 'Knot')
COLORS = ('#28A6A6', '#3B5BA9', '#EEA23A', '#CB4055')


def cache_key(paths, dec_center, dec_width, ra_center, ra_width, r_max):
    files = []
    for path in paths:
        info = path.stat()
        files.append((str(path.resolve()), info.st_size, info.st_mtime_ns))
    payload = json.dumps({
        'files': files, 'dec_center': dec_center, 'dec_width': dec_width,
        'ra_center': ra_center, 'ra_width': ra_width, 'r_max': r_max,
    }, sort_keys=True).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:20]


def select_galaxies(points_path, class_path, summary_path, dec_center, dec_width,
                    ra_center, ra_width, r_max, chunk_size):
    with summary_path.open() as handle:
        summary = json.load(handle)

    if summary.get('catalog') != 'cutsky' or summary.get('config', {}).get('redshift') != 'Z_COSMO':
        raise ValueError('Expected a CutSky run without RSD: {}'.format(summary_path))
    n_real = int(summary['n_real'])
    points = np.load(points_path, mmap_mode='r', allow_pickle=False)
    classes = np.load(class_path, mmap_mode='r', allow_pickle=False)
    point_fields = {'TARGETID', 'ISDATA', 'RA', 'DEC', 'XCART', 'YCART', 'ZCART'}
    class_fields = {'TARGETID', 'ISDATA', 'WEBTYPE'}

    if points.ndim != 1 or points.dtype.names is None or not point_fields.issubset(points.dtype.names):
        raise ValueError('Invalid points array: {}'.format(points_path))
    if classes.ndim != 1 or classes.dtype.names is None or not class_fields.issubset(classes.dtype.names):
        raise ValueError('Invalid classification array: {}'.format(class_path))
    if n_real < 1 or n_real > len(points) or len(classes) != len(points):
        raise ValueError('Inconsistent real-galaxy or classification lengths')

    ra_parts, radius_parts, class_parts = [], [], []
    for start in range(0, n_real, chunk_size):
        stop = min(start + chunk_size, n_real)
        block = points[start:stop]
        delta_ra = (block['RA'].astype(np.float64) - ra_center + 180) % 360 - 180
        mask = (block['ISDATA'] &
                (block['DEC'] >= dec_center - dec_width / 2) &
                (block['DEC'] < dec_center + dec_width / 2) &
                (delta_ra >= -ra_width / 2) & (delta_ra < ra_width / 2))
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        rows = block[indices]
        classified = classes[start + indices]
        if not np.array_equal(rows['TARGETID'], classified['TARGETID']):
            raise ValueError('TARGETID mismatch at rows {}:{}'.format(start, stop))
        if not np.all(classified['ISDATA']):
            raise ValueError('Classification includes random objects')
        radius = np.sqrt(rows['XCART'].astype(np.float64) ** 2 +
                         rows['YCART'].astype(np.float64) ** 2 +
                         rows['ZCART'].astype(np.float64) ** 2)
        radial_mask = np.isfinite(radius) & (radius <= r_max)
        if not radial_mask.any():
            continue
        webtype = classified['WEBTYPE'][radial_mask]
        if np.any(webtype > 3):
            raise ValueError('WEBTYPE outside 0..3 at rows {}:{}'.format(start, stop))
        ra_parts.append(delta_ra[indices][radial_mask].astype(np.float32))
        radius_parts.append(radius[radial_mask].astype(np.float32))
        class_parts.append(webtype.astype(np.uint8))
    if not ra_parts:
        raise ValueError('No real galaxies in the selected wedge')
    return np.concatenate(ra_parts), np.concatenate(radius_parts), np.concatenate(class_parts)


def save_cache(path, delta_ra, radius, webtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.{}.npz'.format(os.getpid()))
    try:
        np.savez_compressed(temporary, delta_ra=delta_ra, radius=radius, webtype=webtype)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def plot(path, delta_ra, radius, webtype, ra_center, ra_width, r_max, point_size, alpha):
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_axes((0.03, 0.07, 0.94, 0.85), projection='polar')
    ax.set_facecolor('white')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(-ra_width / 2)
    ax.set_thetamax(ra_width / 2)
    ax.set_ylim(0, r_max)

    for class_id, color in enumerate(COLORS):
        mask = webtype == class_id
        ax.scatter(np.deg2rad(delta_ra[mask]), radius[mask], s=point_size,
                   c=color, alpha=alpha, linewidths=0, rasterized=True)
    tick_step = 15 if ra_width >= 60 else 10
    offsets = np.arange(-ra_width / 2, ra_width / 2 + 0.01, tick_step)
    if offsets[-1] < ra_width / 2 - 0.01:
        offsets = np.r_[offsets, ra_width / 2]

    labels = [r'${:.0f}^\circ$'.format((ra_center + value) % 360) for value in offsets]
    ax.set_thetagrids(offsets, labels, fontsize=12)
    radial_ticks = np.linspace(0, r_max, 5)[1:]
    ax.set_rticks(radial_ticks)
    ax.set_yticklabels([])
    label_angle = np.deg2rad(ra_width / 2)
    for value in radial_ticks:
        ax.annotate(r'${:g}$'.format(value), xy=(label_angle, value),
                    xytext=(8, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=11)

    ax.grid(color='0.72', linewidth=0.65, alpha=0.7)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.86, r'$\alpha\,(\mathrm{RA})$', ha='center', fontsize=14)
    fig.text(0.73, 0.38, r'$r\,[\mathrm{Mpc}]$',
             ha='center', va='center', rotation=50, fontsize=16)
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=7,
                      markerfacecolor=color, markeredgecolor='none', label=label)
               for label, color in zip(LABELS, COLORS)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.005),
               ncol=4, frameon=False, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=360)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--dec-center', type=float, default=0.0)
    parser.add_argument('--dec-width', type=float, default=6.0)
    parser.add_argument('--ra-center', type=float, default=180.0)
    parser.add_argument('--ra-width', type=float, default=90.0)
    parser.add_argument('--r-max', type=float, default=1600.0)
    parser.add_argument('--chunk-size', type=int, default=1_000_000)
    parser.add_argument('--point-size', type=float, default=0.12)
    parser.add_argument('--alpha', type=float, default=0.65)
    parser.add_argument('--cache-dir', type=Path, default=Path(__file__).with_name('cutsky_wedge_cache'))
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('cutsky_ra_distance_wedge_classified.png'))
    args = parser.parse_args()

    points_path = args.root / 'points' / 'iter_{:03d}.npy'.format(args.iteration)
    class_path = args.root / 'classification' / 'iter_{:03d}.npy'.format(args.iteration)
    summary_path = args.root / 'summary' / 'iter_{:03d}.json'.format(args.iteration)
    key = cache_key((points_path, class_path, summary_path), args.dec_center,
                    args.dec_width, args.ra_center, args.ra_width, args.r_max)
    cache_path = args.cache_dir / 'classified_{}.npz'.format(key)
    if cache_path.exists() and not args.recompute:
        with np.load(cache_path, allow_pickle=False) as saved:
            delta_ra, radius, webtype = (saved[name] for name in
                                         ('delta_ra', 'radius', 'webtype'))
        print('Loaded cache: {}'.format(cache_path))
    else:
        delta_ra, radius, webtype = select_galaxies(
            points_path, class_path, summary_path, args.dec_center, args.dec_width,
            args.ra_center, args.ra_width, args.r_max, args.chunk_size)
        save_cache(cache_path, delta_ra, radius, webtype)
        print('Saved cache: {}'.format(cache_path))

    plot(args.output, delta_ra, radius, webtype, args.ra_center, args.ra_width,
         args.r_max, args.point_size, args.alpha)
    print('Real galaxies: {:,}'.format(len(radius)))
    print('Classes: {}'.format(', '.join(
        '{}={:,}'.format(label, count) for label, count in
        zip(LABELS, np.bincount(webtype, minlength=4)))))
    print('Figure: {} and {}'.format(args.output, args.output.with_suffix('.pdf')))


if __name__ == '__main__':
    main()
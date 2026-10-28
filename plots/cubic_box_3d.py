import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

import matplotlib
import matplotlib.pyplot as pl
matplotlib.use('Agg')
plt.rcParams.update({'text.usetex': True, 'font.size': 11})

ROOT = Path('/pscratch/sd/v/vtorresg/astra-cpp-runs/cubic/BGS_box_ph000')
FIELDS = ('XCART', 'YCART', 'ZCART')


def cache_key(points_path, summary_path, center, width):
    files = []
    for path in (points_path, summary_path):
        info = path.stat()
        files.append((str(path.resolve()), info.st_size, info.st_mtime_ns))
    payload = json.dumps({'files': files, 'center': center, 'width': width},
                         sort_keys=True).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:20]


def select_galaxies(points_path, summary_path, center, width, chunk_size):
    with summary_path.open() as handle:
        summary = json.load(handle)
    if summary.get('catalog') != 'cubic':
        raise ValueError('Expected a cubic catalog: {}'.format(summary_path))
    n_real = int(summary['n_real'])
    points = np.load(points_path, mmap_mode='r', allow_pickle=False)
    required = set(FIELDS) | {'ISDATA'}
    if points.ndim != 1 or points.dtype.names is None or not required.issubset(points.dtype.names):
        raise ValueError('Invalid points array: {}'.format(points_path))
    if n_real < 1 or n_real > len(points):
        raise ValueError('Invalid n_real in {}'.format(summary_path))

    lower = np.asarray(center) - width / 2
    upper = np.asarray(center) + width / 2
    selected = [[] for _ in FIELDS]
    for start in range(0, n_real, chunk_size):
        block = points[start:min(start + chunk_size, n_real)]
        mask = block['ISDATA'].copy()
        for axis, field in enumerate(FIELDS):
            mask &= (block[field] >= lower[axis]) & (block[field] < upper[axis])
        if mask.any():
            for parts, field in zip(selected, FIELDS):
                parts.append(np.asarray(block[field][mask]))
    if not selected[0]:
        raise ValueError('No real galaxies in the selected cube')
    return {axis: np.concatenate(parts) for axis, parts in zip('xyz', selected)}


def save_cache(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.{}.npz'.format(os.getpid()))
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def plot(path, arrays, center, width, elev, azim, point_size, alpha):
    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(arrays['x'], arrays['y'], arrays['z'], c='black', s=point_size,
               alpha=alpha, linewidths=0, depthshade=False, rasterized=True)
    lower = np.asarray(center) - width / 2
    upper = np.asarray(center) + width / 2
    for setter, lo, hi in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lower, upper):
        setter(lo, hi)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(r'$x$ [$h^{-1}$ Mpc]', labelpad=8)
    ax.set_ylabel(r'$y$ [$h^{-1}$ Mpc]', labelpad=8)
    ax.set_zlabel(r'$z$ [$h^{-1}$ Mpc]', labelpad=8)
    # ax.set_title('Cubic box: real galaxies (unclassified)\n'
    #              '{:,} galaxies in a {:.0f} $h^{{-1}}$ Mpc cube'.format(
    #                  len(arrays['x']), width), pad=16)
    ax.tick_params(labelsize=10, pad=2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('0.8')
        axis._axinfo['grid']['color'] = (0.85, 0.85, 0.85, 0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--center-x', type=float, default=1000.0)
    parser.add_argument('--center-y', type=float, default=1000.0)
    parser.add_argument('--center-z', type=float, default=1000.0)
    parser.add_argument('--width', type=float, default=120.0)
    parser.add_argument('--chunk-size', type=int, default=1_000_000)
    parser.add_argument('--elev', type=float, default=25.0)
    parser.add_argument('--azim', type=float, default=-60.0)
    parser.add_argument('--point-size', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--cache-dir', type=Path, default=Path(__file__).with_name('cubic_box_3d_cache'))
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('cubic_box_3d.png'))
    args = parser.parse_args()

    center = (args.center_x, args.center_y, args.center_z)
    points_path = args.root / 'points' / 'iter_{:03d}.npy'.format(args.iteration)
    summary_path = args.root / 'summary' / 'iter_{:03d}.json'.format(args.iteration)

    key = cache_key(points_path, summary_path, center, args.width)
    cache_path = args.cache_dir / 'cube_{}.npz'.format(key)
    if cache_path.exists() and not args.recompute:
        with np.load(cache_path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in 'xyz'}
        print('Loaded cache: {}'.format(cache_path))
    else:
        arrays = select_galaxies(points_path, summary_path, center,
                                  args.width, args.chunk_size)
        save_cache(cache_path, arrays)
        print('Saved cache: {}'.format(cache_path))
    plot(args.output, arrays, center, args.width, args.elev, args.azim,
         args.point_size, args.alpha)
    print('Real galaxies: {:,}'.format(len(arrays['x'])))
    print('Figure: {} and {}'.format(args.output, args.output.with_suffix('.pdf')))


if __name__ == '__main__':
    main()
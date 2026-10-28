import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
plt.rcParams.update({'text.usetex': True})

BASE = Path('/pscratch/sd/v/vtorresg/astra-cpp-runs/cutsky/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000')
FIELDS = ('PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT')
LABELS = ('Void', 'Sheet', 'Filament', 'Knot')
COLORS = ('#3B5BA9', '#28A6A6', '#EEA23A', '#CB4055')
CACHE_VERSION = 1


def input_paths(root):
    return (root / 'points' / 'iter_000.npy', root / 'probability' / 'all_completed.npy')


def cache_key(roots, center, width, depth):
    files = []
    for root in roots:
        for path in input_paths(root):
            info = path.stat()
            files.append((str(path.resolve()), info.st_size, info.st_mtime_ns))
    payload = json.dumps({'version': CACHE_VERSION, 'files': files, 'center': center,
                          'width': width, 'depth': depth}, sort_keys=True).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:20]


def select_galaxies(root, center, width, depth, chunk_size):
    points_path, probability_path = input_paths(root)
    points = np.load(points_path, mmap_mode='r', allow_pickle=False)
    probability = np.load(probability_path, mmap_mode='r', allow_pickle=False)
    required_points = {'TARGETID', 'ISDATA', 'XCART', 'YCART', 'ZCART'}

    if points.ndim != 1 or points.dtype.names is None or not required_points.issubset(points.dtype.names):
        raise ValueError('Invalid points array: {}'.format(points_path))
    required_probability = {'TARGETID', *FIELDS}
    if probability.ndim != 1 or probability.dtype.names is None or not required_probability.issubset(probability.dtype.names):
        raise ValueError('Invalid probability array: {}'.format(probability_path))
    n_real = len(probability)
    if len(points) < n_real:
        raise ValueError('Fewer points than probabilities: {}'.format(root))

    xmin, xmax = center[0] - width / 2, center[0] + width / 2
    ymin, ymax = center[1] - width / 2, center[1] + width / 2
    zmin, zmax = center[2] - depth / 2, center[2] + depth / 2
    selected = {'TARGETID': [], 'X': [], 'Y': [], 'Z': [], 'CLASS': [], 'PMAX': []}
    for start in range(0, n_real, chunk_size):
        stop = min(start + chunk_size, n_real)
        block = points[start:stop]
        mask = ((block['XCART'] >= xmin) & (block['XCART'] < xmax) &
                (block['YCART'] >= ymin) & (block['YCART'] < ymax) &
                (block['ZCART'] >= zmin) & (block['ZCART'] < zmax))
        indices = np.flatnonzero(mask)

        if not len(indices):
            continue
        rows = block[indices]
        probs = probability[start + indices]
        if not np.all(rows['ISDATA']):
            raise ValueError('Selected points include randoms')
        if not np.array_equal(rows['TARGETID'], probs['TARGETID']):
            raise ValueError('TARGETID differs between points and probabilities')
        values = np.column_stack([probs[field] for field in FIELDS])
        if not np.isfinite(values).all():
            raise ValueError('Non-finite ASTRA probabilities')

        classes = np.argmax(values, axis=1)
        selected['TARGETID'].append(np.asarray(rows['TARGETID']))
        selected['X'].append(np.asarray(rows['XCART']))
        selected['Y'].append(np.asarray(rows['YCART']))
        selected['Z'].append(np.asarray(rows['ZCART']))
        selected['CLASS'].append(classes.astype(np.uint8))
        selected['PMAX'].append(values[np.arange(len(values)), classes])
    if not selected['X']:
        raise ValueError('No galaxies found in the requested box: {}'.format(root))
    return {name: np.concatenate(parts) for name, parts in selected.items()}



def match_galaxies(with_root, selected_without):
    points_path, probability_path = input_paths(with_root)
    points = np.load(points_path, mmap_mode='r', allow_pickle=False)
    probability = np.load(probability_path, mmap_mode='r', allow_pickle=False)
    ids = selected_without['without_TARGETID']
    indices = ids - 1  # Cutsky real TARGETID is its one-based catalog row

    if not np.issubdtype(ids.dtype, np.integer) or np.any(indices < 0) or np.any(indices >= len(probability)):
        raise ValueError('Selected TARGETIDs cannot be mapped to real catalog rows')
    rows = points[indices]
    probs = probability[indices]
    if not np.array_equal(rows['TARGETID'], ids) or not np.array_equal(probs['TARGETID'], ids):
        raise ValueError('TARGETID mismatch when matching selected galaxies with RSD')
    if not np.all(rows['ISDATA']):
        raise ValueError('Matched RSD objects include randoms')

    values = np.column_stack([probs[field] for field in FIELDS])
    if not np.isfinite(values).all():
        raise ValueError('Non-finite RSD probabilities in matched galaxies')
    classes = np.argmax(values, axis=1)
    result = {'without_' + name: selected_without['without_' + name]
              for name in ('TARGETID', 'X', 'Y', 'Z', 'CLASS', 'PMAX')}
    result.update({'with_TARGETID': ids,
                   'with_X': np.asarray(rows['XCART']),
                   'with_Y': np.asarray(rows['YCART']),
                   'with_Z': np.asarray(rows['ZCART']),
                   'with_CLASS': classes.astype(np.uint8),
                   'with_PMAX': values[np.arange(len(values)), classes]})
    return result


def save_cache(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.{}.npz'.format(os.getpid()))
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def plot(path, arrays, center, width, depth, matched=False):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11.5, 5.8), constrained_layout=True)
    box_bounds = (center[0] - width / 2, center[0] + width / 2,
                  center[1] - width / 2, center[1] + width / 2)
    bounds = box_bounds
    if matched:
        # Keep all matched galaxies visible, including those beyond the XY box.
        x = arrays['with_X']
        y = arrays['with_Y']
        bounds = (min(box_bounds[0], float(x.min())) - 2,
                  max(box_bounds[1], float(x.max())) + 2,
                  min(box_bounds[2], float(y.min())) - 2,
                  max(box_bounds[3], float(y.max())) + 2)
    for ax, tag, title in ((axes[0], 'without', 'Without RSD'), (axes[1], 'with', 'With RSD')):
        x, y, classes = (arrays[tag + '_' + field] for field in ('X', 'Y', 'CLASS'))
        for class_id, color in enumerate(COLORS):
            mask = classes == class_id
            ax.scatter(x[mask], y[mask], s=8, c=color, alpha=0.85, linewidths=0, rasterized=True)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect('equal', adjustable='box')

        if matched:
            ax.add_patch(Rectangle((box_bounds[0], box_bounds[2]), width, width,
                                   fill=False, edgecolor='0.45', linestyle='--', linewidth=0.8))
        ax.set_title('{} (N = {:,})'.format(title, len(x)))
        ax.set_xlabel(r'$x$ [Mpc]')
        ax.grid(alpha=0.12, linewidth=0.5)

    axes[0].set_ylabel(r'$y$ [Mpc]')
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=7,
                      markerfacecolor=color, markeredgecolor='none', label=label)
               for color, label in zip(COLORS, LABELS)]
    fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False, fontsize=15)
    if matched:
        title = r'Same galaxies selected without RSD, $z_{\mathrm{cart}} \in [%.0f, %.0f)$ Mpc'
    else:
        title = r'ASTRA environment (max probability) for CutSky, $z_{\mathrm{cart}} \in [%.0f, %.0f)$ Mpc'
    fig.suptitle(title % (center[2] - depth / 2, center[2] + depth / 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)



def plot_changes(path, arrays, center, width, depth):
    changed = arrays['without_CLASS'] != arrays['with_CLASS']
    n_changed = int(np.count_nonzero(changed))
    n_total = len(changed)
    box_bounds = (center[0] - width / 2, center[0] + width / 2,
                  center[1] - width / 2, center[1] + width / 2)

    x_rsd, y_rsd = arrays['with_X'], arrays['with_Y']
    bounds = (min(box_bounds[0], float(x_rsd.min())) - 2,
              max(box_bounds[1], float(x_rsd.max())) + 2,
              min(box_bounds[2], float(y_rsd.min())) - 2,
              max(box_bounds[3], float(y_rsd.max())) + 2)

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11.5, 5.8), constrained_layout=True)
    for ax, tag, title in ((axes[0], 'without', 'Without RSD'), (axes[1], 'with', 'With RSD')):
        x, y = arrays[tag + '_X'], arrays[tag + '_Y']
        ax.scatter(x[~changed], y[~changed], s=6, c='#A0A0A0', alpha=0.55,
                   linewidths=0, rasterized=True)
        ax.scatter(x[changed], y[changed], s=9, c='#D62728', alpha=0.9,
                   linewidths=0, rasterized=True)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect('equal', adjustable='box')
        ax.add_patch(Rectangle((box_bounds[0], box_bounds[2]), width, width,
                               fill=False, edgecolor='0.45', linestyle='--', linewidth=0.8))
        ax.set_title('{} (N = {:,})'.format(title, n_total))
        ax.set_xlabel(r'$x$ [Mpc]')
        ax.grid(alpha=0.12, linewidth=0.5)

    axes[0].set_ylabel(r'$y$ [Mpc]')
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=7,
                      markerfacecolor=color, markeredgecolor='none', label=label)
               for color, label in (('#A0A0A0', 'Unchanged ({:,})'.format(n_total - n_changed)),
                                    ('#D62728', r'Changed ({:,}; {:.1f}\%)'.format(n_changed, 100 * n_changed / n_total)))]

    fig.legend(handles=handles, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle(r'ASTRA environment changes, no-RSD selection '
                 r'$z_{\mathrm{cart}} \in [%.0f, %.0f)$ Mpc' % (center[2] - depth / 2, center[2] + depth / 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)



def plot_transitions(path, arrays, center, width, depth):
    before = arrays['without_CLASS']
    after = arrays['with_CLASS']
    code = 4 * before.astype(np.int16) + after.astype(np.int16)
    transitions = ((0, 1, '#377EB8', r'Void $\to$ Sheet'),
                   (1, 2, '#2CA25F', r'Sheet $\to$ Filament'),
                   (2, 3, '#D94841', r'Filament $\to$ Knot'),
                   (3, 2, '#8E63B8', r'Knot $\to$ Filament'))

    selected_codes = [4 * source + target for source, target, _, _ in transitions]
    unchanged = before == after
    other_changed = (~unchanged) & (~np.isin(code, selected_codes))
    box_bounds = (center[0] - width / 2, center[0] + width / 2,
                  center[1] - width / 2, center[1] + width / 2)
    x_rsd, y_rsd = arrays['with_X'], arrays['with_Y']
    bounds = (min(box_bounds[0], float(x_rsd.min())) - 2,
              max(box_bounds[1], float(x_rsd.max())) + 2,
              min(box_bounds[2], float(y_rsd.min())) - 2,
              max(box_bounds[3], float(y_rsd.max())) + 2)

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11.5, 6.2), constrained_layout=True)
    for ax, tag, title in ((axes[0], 'without', 'Without RSD'), (axes[1], 'with', 'With RSD')):
        x, y = arrays[tag + '_X'], arrays[tag + '_Y']
        ax.scatter(x[unchanged], y[unchanged], s=5, c='#C9C9C9', alpha=0.45,
                   linewidths=0, rasterized=True)
        ax.scatter(x[other_changed], y[other_changed], s=7, c='#6F6F6F', alpha=0.65,
                   linewidths=0, rasterized=True)
        for source, target, color, _ in transitions:
            mask = code == 4 * source + target
            ax.scatter(x[mask], y[mask], s=10, c=color, alpha=0.9,
                       linewidths=0, rasterized=True)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect('equal', adjustable='box')
        ax.add_patch(Rectangle((box_bounds[0], box_bounds[2]), width, width,
                               fill=False, edgecolor='0.45', linestyle='--', linewidth=0.8))
        ax.set_title('{} (N = {:,})'.format(title, len(before)), y=1.02)
        ax.set_xlabel(r'$x$ [Mpc]')
        ax.grid(alpha=0.12, linewidth=0.5)

    axes[0].set_ylabel(r'$y$ [Mpc]')
    legend_entries = [('#C9C9C9', 'Unchanged ({:,})'.format(np.count_nonzero(unchanged))),
                      ('#6F6F6F', 'Other changes ({:,})'.format(np.count_nonzero(other_changed)))]
    legend_entries += [(color, '{} ({:,})'.format(label, np.count_nonzero(code == 4 * source + target)))
                       for source, target, color, label in transitions]
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=7,
                      markerfacecolor=color, markeredgecolor='none', label=label) for color, label in legend_entries]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False, fontsize=15)

    fig.get_layout_engine().set(h_pad=0.07)
    fig.suptitle(r'ASTRA environment transitions, no-RSD selection $z_{\mathrm{cart}} \in [%.0f, %.0f)$ Mpc' % (center[2] - depth / 2, center[2] + depth / 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    # fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)



def displacement_data(arrays):
    '''Return 3D comoving displacement and class-change flag for matched IDs.'''
    if not np.array_equal(arrays['without_TARGETID'], arrays['with_TARGETID']):
        raise ValueError('Cannot calculate displacement for unmatched TARGETIDs')
    components = [
        arrays['with_' + name].astype(np.float64) -
        arrays['without_' + name].astype(np.float64)
        for name in ('X', 'Y', 'Z')
    ]
    displacement = np.sqrt(sum(component ** 2 for component in components))
    changed = arrays['without_CLASS'] != arrays['with_CLASS']
    return displacement, changed


def write_displacement_csv(path, arrays, displacement, changed):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['TARGETID', 'delta_s_mpc', 'class_without_rsd', 'class_with_rsd', 'changed'])
        for targetid, delta, before, after, did_change in zip(
            arrays['without_TARGETID'], displacement,
            arrays['without_CLASS'], arrays['with_CLASS'], changed
        ):
            writer.writerow([
                int(targetid), float(delta), LABELS[int(before)],
                LABELS[int(after)], int(did_change),
            ])


def plot_displacement(path, displacement, changed):
    unchanged = ~changed
    upper = float(np.ceil(displacement.max()))
    edges = np.linspace(0, upper, 61)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for mask, color, label in ((unchanged, '#777777', 'Unchanged'), (changed, '#D62728', 'Changed')):
        counts = np.histogram(displacement[mask], bins=edges)[0]
        density = counts / (mask.sum() * np.diff(edges))
        median = np.median(displacement[mask])
        axes[0].stairs(density, edges, color=color, linewidth=1.9, label='{} (median {:.2f} Mpc)'.format(label, median))

    axes[0].set_xlim(0, upper)
    axes[0].set_ylim(bottom=0)
    axes[0].set_xlabel(r'$\Delta s=\|\mathbf{x}_{\mathrm{RSD}}-\mathbf{x}_{\mathrm{no\,RSD}}\|$ [Mpc]')
    axes[0].set_ylabel('Probability density')
    axes[0].set_title('Comoving displacement')
    axes[0].legend(frameon=False)
    axes[0].grid(axis='y', alpha=0.2)

    quantile_edges = np.quantile(displacement, np.linspace(0, 1, 13))
    if np.any(np.diff(quantile_edges) <= 0):
        raise ValueError('Duplicate displacement quantiles prevent equal-count bins')
    all_counts = np.histogram(displacement, bins=quantile_edges)[0]
    changed_counts = np.histogram(displacement[changed], bins=quantile_edges)[0]
    bin_centers = np.array([displacement[(displacement >= lo) & (displacement < hi if i < len(all_counts) - 1 else displacement <= hi)].mean()
                            for i, (lo, hi) in enumerate(zip(quantile_edges[:-1], quantile_edges[1:]))])

    fraction = changed_counts / all_counts
    z_score = 1.96
    denominator = 1 + z_score ** 2 / all_counts
    wilson_center = (fraction + z_score ** 2 / (2 * all_counts)) / denominator
    wilson_half = z_score * np.sqrt(fraction * (1 - fraction) / all_counts + z_score ** 2 / (4 * all_counts ** 2)) / denominator
    low = wilson_center - wilson_half
    high = wilson_center + wilson_half

    axes[1].errorbar(bin_centers, fraction, yerr=[fraction - low, high - fraction],
                     fmt='o-', color='#D62728', capsize=2.5, markersize=4.5,
                     linewidth=1.3, label='Change fraction (95\\% CI)')
    axes[1].axhline(changed.mean(), color='0.45', linestyle='--', linewidth=1,
                    label='Overall change fraction')
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel(r'$\Delta s$ [Mpc]')
    axes[1].set_ylabel('Fraction changing class')
    axes[1].set_title('Class changes vs. displacement')
    axes[1].legend(frameon=False)
    axes[1].grid(axis='y', alpha=0.2)

    fig.suptitle('ASTRA class changes and comoving displacement')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-rsd-root', type=Path, default=BASE / 'z_cosmo')
    parser.add_argument('--rsd-root', type=Path, default=BASE / 'z')
    parser.add_argument('--center-x', type=float, default=50.0)
    parser.add_argument('--center-y', type=float, default=150.0)
    parser.add_argument('--center-z', type=float, default=705.0)
    parser.add_argument('--width', type=float, default=200.0)
    parser.add_argument('--depth', type=float, default=40.0)
    parser.add_argument('--chunk-size', type=int, default=1_000_000)
    parser.add_argument('--cache-dir', type=Path, default=Path(__file__).with_name('sky_slice_cache'))
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--matched-output', type=Path)
    parser.add_argument('--changes-output', type=Path)
    parser.add_argument('--transitions-output', type=Path)
    parser.add_argument('--displacement-output', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('cutsky_environment_slice.png'))
    args = parser.parse_args()

    if args.width <= 0 or args.depth <= 0 or args.chunk_size < 1:
        parser.error('Width, depth, and chunk size must be positive')
    center = (args.center_x, args.center_y, args.center_z)
    roots = (args.no_rsd_root, args.rsd_root)
    key = cache_key(roots, center, args.width, args.depth)
    cache_path = args.cache_dir / 'slice_{}.npz'.format(key)
    if cache_path.exists() and not args.recompute:
        with np.load(cache_path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        print('Loaded cache: {}'.format(cache_path))
    else:
        arrays = {}
        for tag, root in (('without', args.no_rsd_root), ('with', args.rsd_root)):
            selected = select_galaxies(root, center, args.width, args.depth, args.chunk_size)
            arrays.update({tag + '_' + name: values for name, values in selected.items()})
        save_cache(cache_path, arrays)
        print('Saved cache: {}'.format(cache_path))
    for tag in ('without', 'with'):
        classes = arrays[tag + '_CLASS']
        counts = np.bincount(classes, minlength=4)
        print('{} RSD: {:,} galaxies; {}'.format(tag, len(classes), ', '.join('{}={:,}'.format(label, count) for label, count in zip(LABELS, counts))))

    plot(args.output, arrays, center, args.width, args.depth)
    print('Figure: {} and {}'.format(args.output, args.output.with_suffix('.pdf')))

    matched_output = args.matched_output or args.output.with_name(args.output.stem + '_matched' + args.output.suffix)

    matched_cache = args.cache_dir / 'matched_{}.npz'.format(key)
    if matched_cache.exists() and not args.recompute:
        with np.load(matched_cache, allow_pickle=False) as saved:
            matched_arrays = {name: saved[name] for name in saved.files}
        print('Loaded matched cache: {}'.format(matched_cache))
    else:
        matched_arrays = match_galaxies(args.rsd_root, arrays)
        save_cache(matched_cache, matched_arrays)
        print('Saved matched cache: {}'.format(matched_cache))
    if not np.array_equal(matched_arrays['without_TARGETID'], matched_arrays['with_TARGETID']):
        raise ValueError('The two matched panels have different TARGETIDs')

    n_matched = len(matched_arrays['without_TARGETID'])
    changed = np.count_nonzero(matched_arrays['without_CLASS'] != matched_arrays['with_CLASS'])
    z_rsd = matched_arrays['with_Z']
    outside_z = np.count_nonzero((z_rsd < center[2] - args.depth / 2) |
                                 (z_rsd >= center[2] + args.depth / 2))

    print('Matched galaxies: {:,}; changed class: {:,}; outside RSD slab: {:,}'.format(n_matched, changed, outside_z))
    plot(matched_output, matched_arrays, center, args.width, args.depth, matched=True)
    print('Matched figure: {} and {}'.format(matched_output, matched_output.with_suffix('.pdf')))

    changes_output = args.changes_output or args.output.with_name(args.output.stem + '_changes' + args.output.suffix)
    plot_changes(changes_output, matched_arrays, center, args.width, args.depth)
    print('Changes figure: {} and {}'.format(changes_output, changes_output.with_suffix('.pdf')))

    transitions_output = args.transitions_output or args.output.with_name(args.output.stem + '_transitions' + args.output.suffix)
    plot_transitions(transitions_output, matched_arrays, center, args.width, args.depth)
    print('Transitions figure: {} and {}'.format(transitions_output, transitions_output.with_suffix('.pdf')))

    displacement_output = args.displacement_output or args.output.with_name(args.output.stem + '_displacement' + args.output.suffix)
    displacement, changed_flags = displacement_data(matched_arrays)
    write_displacement_csv(displacement_output.with_suffix('.csv'), matched_arrays, displacement, changed_flags)
    plot_displacement(displacement_output, displacement, changed_flags)
    print('Displacement figure: {} and {}'.format(displacement_output, displacement_output.with_suffix('.pdf')))


if __name__ == '__main__':
    main()

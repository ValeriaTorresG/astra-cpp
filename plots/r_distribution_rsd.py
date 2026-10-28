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
plt.rcParams.update({'text.usetex': True, 'font.size': 12})

BASE = Path('/pscratch/sd/v/vtorresg/astra-cpp-runs/cutsky/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000')
THRESHOLDS = (-0.25, 0.25, 0.65)
CACHE_VERSION = 1
SERIES = ('without_rsd_real', 'with_rsd_real', 'without_rsd_random', 'with_rsd_random')


def completed_iterations(root):
    iterations = set()
    for path in (root / 'summary').glob('iter_*.json'):
        with path.open() as handle:
            summary = json.load(handle)
        if summary.get('status') == 'complete':
            iterations.add(int(path.stem.split('_')[1]))
    return iterations


def open_iteration(root, iteration):
    path = root / 'classification' / 'iter_{:03d}.npy'.format(iteration)
    summary_path = root / 'summary' / 'iter_{:03d}.json'.format(iteration)
    with summary_path.open() as handle:
        summary = json.load(handle)
    n_real = int(summary['n_real'])
    n_random = int(summary['n_random'])
    data = np.load(path, mmap_mode='r', allow_pickle=False)
    required = {'TARGETID', 'ISDATA', 'R'}
    if data.ndim != 1 or data.dtype.names is None or not required.issubset(data.dtype.names):
        raise ValueError('Invalid classification array: {}'.format(path))
    if n_real < 1 or n_random < 1 or len(data) != n_real + n_random:
        raise ValueError('Invalid sample sizes in {}'.format(summary_path))
    return data, n_real, n_random


def count_histograms(no_rsd, rsd, n_real, n_random, edges, chunk_size):
    counts = {name: np.zeros(len(edges) - 1, dtype=np.int64) for name in SERIES}
    for sample, offset, size, is_data in (('real', 0, n_real, True), ('random', n_real, n_random, False)):
        for start in range(offset, offset + size, chunk_size):
            end = min(start + chunk_size, offset + size)
            a = no_rsd[start:end]
            b = rsd[start:end]

            if not (np.all(a['ISDATA'] == is_data) and np.all(b['ISDATA'] == is_data)):
                raise ValueError('Unexpected ISDATA values in rows {}:{}'.format(start, end))
            if not np.array_equal(a['TARGETID'], b['TARGETID']):
                raise ValueError('TARGETID differs in rows {}:{}'.format(start, end))

            for tag, rows in (('without_rsd', a), ('with_rsd', b)):
                values = rows['R']
                if not np.isfinite(values).all():
                    raise ValueError('Non-finite R values in rows {}:{}'.format(start, end))
                if np.any((values < -1) | (values > 1)):
                    raise ValueError('R values outside [-1, 1] in rows {}:{}'.format(start, end))
                counts[tag + '_' + sample] += np.histogram(values, bins=edges)[0]

        for tag in ('without_rsd', 'with_rsd'):
            if counts[tag + '_' + sample].sum() != size:
                raise ValueError('Missing {} objects in the histogram'.format(sample))
    return counts



def cache_key(no_rsd_root, rsd_root, iterations, bins):
    sources = []
    for root in (no_rsd_root, rsd_root):
        for iteration in iterations:
            for subdir, suffix in (('summary', '.json'), ('classification', '.npy')):
                path = root / subdir / 'iter_{:03d}{}'.format(iteration, suffix)
                info = path.stat()
                sources.append((str(path.resolve()), info.st_size, info.st_mtime_ns))
    payload = json.dumps({'version': CACHE_VERSION, 'bins': bins, 'sources': sources},
                         sort_keys=True).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:20]


def save_cache(path, edges, statistics, sample_sizes):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {'edges': edges, 'sample_sizes': np.asarray(sample_sizes)}
    for name in SERIES:
        for metric in ('mean', 'std'):
            arrays[name + '_' + metric] = statistics[name][metric]
    temporary = path.with_name(path.stem + '.tmp.{}.npz'.format(os.getpid()))
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def load_cache(path, bins):
    with np.load(path, allow_pickle=False) as saved:
        edges = saved['edges']
        sample_sizes = tuple(int(value) for value in saved['sample_sizes'])
        statistics = {name: {metric: saved[name + '_' + metric] for metric in ('mean', 'std')}
                      for name in SERIES}

    if edges.shape != (bins + 1,) or len(sample_sizes) != 2:
        raise ValueError('Invalid histogram cache: {}'.format(path))
    if not np.array_equal(edges, np.linspace(-1, 1, bins + 1)):
        raise ValueError('Wrong bin edges in cache: {}'.format(path))
    if any(values[metric].shape != (bins,) for values in statistics.values() for metric in ('mean', 'std')):
        raise ValueError('Invalid PDF arrays in cache: {}'.format(path))
    return edges, statistics, sample_sizes


def write_csv(path, edges, statistics):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['r_left', 'r_right'] + [name + '_' + metric + '_pdf' for name in SERIES for metric in ('mean', 'std')])
        for i in range(len(edges) - 1):
            writer.writerow([edges[i], edges[i + 1]] + [statistics[name][metric][i] for name in SERIES for metric in ('mean', 'std')])


def plot(path, edges, statistics, n_iterations):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4.6), constrained_layout=True)
    styles = (('without_rsd', '#eea33a', 'solid', 'Without RSD'),
              ('with_rsd', '#29a6a6', 'solid', 'With RSD'))

    for ax, sample, panel_title in zip(axes, ('real', 'random'), ('Real objects', 'Random objects')):
        for tag, color, linestyle, label in styles:
            values = statistics[tag + '_' + sample]
            mean, std = values['mean'], values['std']
            lower = np.maximum(mean - std, 0)
            upper = mean + std

            ax.fill_between(edges, np.r_[lower, lower[-1]], np.r_[upper, upper[-1]],
                            step='post', color=color, alpha=0.12, linewidth=0)
            ax.stairs(mean, edges, color=color, linestyle=linestyle,
                      linewidth=2.1, label=label, zorder=3)
        for threshold in THRESHOLDS:
            ax.axvline(threshold, color='0.55', linestyle='--', linewidth=1.,
                       alpha=1., zorder=1)
        ax.set_xlim(-1, 1)
        ax.set_ylim(bottom=0)
        ax.set_title(panel_title, fontsize=13, pad=8)
        ax.set_xlabel(r'$r=(N_{\mathrm{data}}-N_{\mathrm{rand}})/(N_{\mathrm{data}}+N_{\mathrm{rand}})$',
                      labelpad=10, fontsize=13)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(axis='y', color='0.85', linewidth=0.7)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('PDF', labelpad=10, fontsize=13)
    fig.suptitle(r'ASTRA: PDF of $r$ across {} iterations'.format(n_iterations))
    # fig.text(0.5, 0.01, r'$r=(N_{\mathrm{data}}-N_{\mathrm{rand}})/(N_{\mathrm{data}}+N_{\mathrm{rand}})$'
    #          r'  \quad Curves: mean; shading: $\pm 1$ standard deviation', ha='center', fontsize=9, color='0.3')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=360)
    # fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-rsd-root', type=Path, default=BASE / 'z_cosmo')
    parser.add_argument('--rsd-root', type=Path, default=BASE / 'z')
    parser.add_argument('--bins', type=int, default=40)
    parser.add_argument('--chunk-size', type=int, default=500_000)
    parser.add_argument('--cache-dir', type=Path, default=Path(__file__).with_name('r_distribution_cache'))
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('r_distribution_rsd.png'))
    args = parser.parse_args()

    no_iterations = completed_iterations(args.no_rsd_root)
    yes_iterations = completed_iterations(args.rsd_root)
    if not no_iterations or no_iterations != yes_iterations:
        raise ValueError('The two runs must have the same nonempty set of completed iterations: without RSD={}, with RSD={}'.format(sorted(no_iterations), sorted(yes_iterations)))
    iterations = sorted(no_iterations)
    key = cache_key(args.no_rsd_root, args.rsd_root, iterations, args.bins)

    cache_path = args.cache_dir / 'pdf_{}.npz'.format(key)
    if cache_path.exists() and not args.recompute:
        edges, statistics, sample_sizes = load_cache(cache_path, args.bins)
        print('Loaded cached PDFs: {}'.format(cache_path))
    else:
        edges = np.linspace(-1, 1, args.bins + 1)
        widths = np.diff(edges)
        densities = {name: [] for name in SERIES}
        sample_sizes = None
        for iteration in iterations:
            no_rsd, n_real, n_random = open_iteration(args.no_rsd_root, iteration)
            rsd, n_real_rsd, n_random_rsd = open_iteration(args.rsd_root, iteration)
            if (n_real, n_random) != (n_real_rsd, n_random_rsd):
                raise ValueError('Different sample sizes in iteration {}'.format(iteration))

            if sample_sizes is None:
                sample_sizes = (n_real, n_random)
            elif sample_sizes != (n_real, n_random):
                raise ValueError('Sample sizes change between iterations')

            counts = count_histograms(no_rsd, rsd, n_real, n_random, edges, args.chunk_size)
            for name in SERIES:
                n_sample = n_real if name.endswith('_real') else n_random
                densities[name].append(counts[name] / (n_sample * widths))
            print('Processed iteration {:03d}'.format(iteration), flush=True)
            del no_rsd, rsd

        statistics = {}
        for name in SERIES:
            values = np.asarray(densities[name])
            statistics[name] = {'mean': values.mean(axis=0), 'std': values.std(axis=0)}
        save_cache(cache_path, edges, statistics, sample_sizes)
        print('Saved cached PDFs: {}'.format(cache_path))

    csv_path = args.output.with_suffix('.csv')
    write_csv(csv_path, edges, statistics)
    plot(args.output, edges, statistics, len(iterations))
    print('Completed iterations: {}'.format(len(iterations)))
    print('Real / random objects per iteration: {:,} / {:,}'.format(*sample_sizes))
    print('Figure: {} and {}'.format(args.output, args.output.with_suffix('.pdf')))
    print('PDF data: {}'.format(csv_path))


if __name__ == '__main__':
    main()
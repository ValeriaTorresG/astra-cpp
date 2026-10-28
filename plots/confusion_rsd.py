import argparse
import csv
from pathlib import Path
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.rcParams.update({'text.usetex': True, 'font.size': 15})

BASE = Path('/pscratch/sd/v/vtorresg/astra-cpp-runs/cutsky/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000')
FIELDS = ('PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT')
LABELS = ('Void', 'Sheet', 'Filament', 'Knot')


def load_probabilities(path):
    data = np.load(path, mmap_mode='r', allow_pickle=False)
    if data.ndim != 1 or data.dtype.names is None:
        raise ValueError('Expected a 1D structured array: {}'.format(path))
    missing = set(('TARGETID',) + FIELDS) - set(data.dtype.names)
    if missing:
        raise ValueError('Missing columns {} in {}'.format(sorted(missing), path))
    return data


def confusion_matrix(no_rsd, rsd, chunk_size):
    if len(no_rsd) != len(rsd):
        raise ValueError('The files have different numbers of objects')
    matrix = np.zeros((4, 4), dtype=np.int64)
    for start in range(0, len(no_rsd), chunk_size):
        end = min(start + chunk_size, len(no_rsd))
        a = no_rsd[start:end]
        b = rsd[start:end]
        if not np.array_equal(a['TARGETID'], b['TARGETID']):
            raise ValueError('TARGETID differs between z_cosmo and z in rows {}:{}; align the catalogs before comparing them'.format(start, end))
        pa = np.column_stack([a[field] for field in FIELDS])
        pb = np.column_stack([b[field] for field in FIELDS])
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            raise ValueError('Non-finite probabilities in rows {}:{}'.format(start, end))
        class_a = np.argmax(pa, axis=1)
        class_b = np.argmax(pb, axis=1)
        matrix += np.bincount(4 * class_a + class_b, minlength=16).reshape(4, 4)
    return matrix


def write_csv(path, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['without_rsd / with_rsd'] + list(LABELS))
        for label, row in zip(LABELS, matrix):
            writer.writerow([label] + [int(value) for value in row])


def plot_matrix(path, matrix):
    totals = matrix.sum(axis=1, keepdims=True)
    fractions = np.divide(matrix, totals, out=np.zeros((4, 4), dtype=float), where=totals != 0)

    fig, ax = plt.subplots(figsize=(8, 6.8), constrained_layout=True)
    im = ax.imshow(fractions, cmap='coolwarm', vmin=0, vmax=1)

    ax.set_xticks(range(4))
    ax.set_xticklabels(LABELS, fontsize=15)
    ax.set_yticks(range(4))
    ax.set_yticklabels(LABELS, fontsize=15)
    ax.set_xlabel('With RSD', fontsize=15, labelpad=10)

    ax.set_ylabel('Without RSD', fontsize=15, labelpad=10)
    ax.set_title('Astra: Classification changes due to RSD', fontsize=15, y=1.02)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, '{:.1f}\\%\n{:,.0f}'.format(100 * fractions[i, j], matrix[i, j]),
                    ha='center', va='center', color='black' if fractions[i, j] > 0.2 else 'white', fontsize=13)
    # fig.colorbar(im, ax=ax, )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label='Fraction within the no-RSD class',fontsize=14, labelpad=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-rsd', type=Path, default=BASE / 'z_cosmo/probability/all_completed.npy')
    parser.add_argument('--rsd', type=Path, default=BASE / 'z/probability/all_completed.npy')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('confusion_rsd.png'))
    parser.add_argument('--chunk-size', type=int, default=500_000)
    parser.add_argument('--csv-only', action='store_true')
    args = parser.parse_args()

    no_rsd = load_probabilities(args.no_rsd)
    rsd = load_probabilities(args.rsd)
    matrix = confusion_matrix(no_rsd, rsd, args.chunk_size)
    if matrix.sum() == 0:
        raise ValueError('The files contain no objects')
    csv_path = args.output.with_suffix('.csv')
    write_csv(csv_path, matrix)
    print('Objects compared: {:,}'.format(matrix.sum()))
    print('Same class: {:.2%}'.format(np.trace(matrix) / matrix.sum()))
    print('Count matrix: {}'.format(csv_path))
    if not args.csv_only:
        plot_matrix(args.output, matrix)
        print('Figure: {}'.format(args.output))


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
parallel triangulation scaling on a realistic cut-sky mock subsample.

combine n/2 real galaxies sampled from the full catalog with n/2 randoms from
a few seed files. measure triangulation and neighbor counting for each thread
count in a separate process (with a timeout in case insertion enters livelock).

    python benchmarks/bench_triangulation.py --n 30000000 --threads 32 64 128
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from astra_cpp import _core  # noqa: E402
from astra_cpp.config import CUTSKY_DIR, DEFAULT_CUTSKY_REAL  # noqa: E402

WORKER = '''
import sys, time, numpy as np
sys.path.insert(0, {python!r})
from astra_cpp import _core
x = np.load({points!r}); is_data = np.arange(len(x)) < len(x) // 2
t = time.perf_counter(); d = _core.Delaunay3(x, threads={threads}, lock_grid_cells={grid})
tb = time.perf_counter() - t
t = time.perf_counter(); d.neighbor_counts(is_data, threads={threads}); tc = time.perf_counter() - t
print(f'{{tb:.2f}} {{tc:.2f}}')
'''


def build_sample(n, seeds, threads, path):
    real = _core.FitsTable(str(DEFAULT_CUTSKY_REAL))
    rans = [_core.FitsTable(str(CUTSKY_DIR / f'random_S{s}_1X.fits')) for s in seeds]
    half = n // 2
    rows_real = _core.sample_pool([real.nrows], half, 1)[0]
    rows_rand = _core.sample_pool([t.nrows for t in rans], half, 2)
    cols = [np.empty(2 * half) for _ in range(3)]
    requests, offset = [(real, rows_real, 0)], half
    for table, rows in zip(rans, rows_rand):
        requests.append((table, rows, offset))
        offset += len(rows)
    _core.read_columns(requests, ['RA', 'DEC', 'Z'], cols, threads=threads)
    np.save(path, _core.sky_to_cartesian(*cols, _core.FlatLambdaCDM.planck18(), threads=threads))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=6_000_000)
    ap.add_argument('--threads', type=int, nargs='+', default=[16, 32, 64, 128])
    ap.add_argument('--grid', type=int, nargs='+', default=[0], help='0 = automatic grid size')
    ap.add_argument('--random-seeds', type=int, nargs='+', default=[100, 200, 300])
    ap.add_argument('--timeout', type=float, default=300)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
        points = str(Path(tmp) / 'points.npy')
        t = time.perf_counter()
        build_sample(args.n, args.random_seeds, max(args.threads), points)
        print(f'sample of {args.n} points ready in {time.perf_counter() - t:.1f} s '
              f'(automatic grid { _core.auto_lock_grid_cells(args.n)}^3)', flush=True)
        for grid in args.grid:
            for threads in args.threads:
                code = WORKER.format(python=str(Path(__file__).resolve().parents[1] / 'python'),
                                     points=points, threads=threads, grid=grid)
                try:
                    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                         timeout=args.timeout, check=True).stdout.split()
                    print(f'threads={threads:4d} grid={grid or "auto":>5} triang={out[0]:>8} s '
                          f'neighbors={out[1]:>7} s', flush=True)
                except subprocess.TimeoutExpired:
                    print(f'threads={threads:4d} grid={grid or "auto":>5} TIMEOUT (>{args.timeout:.0f} s)', flush=True)


if __name__ == '__main__':
    main()

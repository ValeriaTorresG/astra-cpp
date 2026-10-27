"""run outputs: paths, atomic .npy/.json writes, and aggregation.

structure of `run_dir`:
    points/iter_XXX.npy          TARGETID, ISDATA, [RA, DEC, Z], XCART, YCART, ZCART
    classification/iter_XXX.npy  TARGETID, ISDATA, NDATA, NRAND, R, WEBTYPE
    summary/iter_XXX.json        configuration, counts, timings, and memory
    logs/iter_XXX.log
    probability/all_completed.npy  TARGETID, PVOID, PSHEET, PFILAMENT, PKNOT (real objects only)
the first n_real rows of points and classification are always real objects.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from . import _core

CLASS_DTYPE = np.dtype([('TARGETID', '<i8'), ('ISDATA', '?'), ('NDATA', '<i4'),
                        ('NRAND', '<i4'), ('R', '<f4'), ('WEBTYPE', 'u1')])
PROB_FIELDS = ('PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT')
PROB_DTYPE = np.dtype([('TARGETID', '<i8')] + [(f, '<f4') for f in PROB_FIELDS])
CHUNK_ROWS = 4_000_000


class RunPaths:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _iter(self, kind: str, iteration: int, ext: str) -> Path:
        return self.root / kind / f'iter_{iteration:03d}.{ext}'

    def points(self, iteration: int) -> Path:
        return self._iter('points', iteration, 'npy')

    def classification(self, iteration: int) -> Path:
        return self._iter('classification', iteration, 'npy')

    def summary(self, iteration: int) -> Path:
        return self._iter('summary', iteration, 'json')

    def log(self, iteration: int) -> Path:
        return self._iter('logs', iteration, 'log')

    def probability(self) -> Path:
        return self.root / 'probability' / 'all_completed.npy'

    def make_dirs(self) -> None:
        for kind in ('points', 'classification', 'summary', 'logs', 'probability'):
            (self.root / kind).mkdir(parents=True, exist_ok=True)


def points_dtype(extra: list[str]) -> np.dtype:
    fields = [('TARGETID', '<i8'), ('ISDATA', '?')]
    fields += [(name, '<f4') for name in extra]
    fields += [('XCART', '<f4'), ('YCART', '<f4'), ('ZCART', '<f4')]
    return np.dtype(fields)


def _tmp(path: Path) -> Path:
    return path.with_name(f'{path.name}.tmp.{os.getpid()}')


def write_structured(path: Path, columns: dict[str, np.ndarray], dtype: np.dtype) -> None:
    """write columns as a structured .npy and publish it atomically.

    assemble each block in memory and write it contiguously: traverse the
    file only once and sequentially.
    """
    n = len(next(iter(columns.values())))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    header = {'descr': np.lib.format.dtype_to_descr(dtype), 'fortran_order': False, 'shape': (n,)}
    buf = np.empty(min(n, CHUNK_ROWS), dtype=dtype)
    try:
        with open(tmp, 'wb') as fh:
            np.lib.format.write_array_header_2_0(fh, header)
            for start in range(0, n, CHUNK_ROWS):
                end = min(n, start + CHUNK_ROWS)
                block = buf[:end - start]
                for name in dtype.names:
                    block[name] = columns[name][start:end]
                block.tofile(fh)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, sort_keys=True, default=str)
        fh.write('\n')
    os.replace(tmp, path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def valid_npy(path: Path, rows: int, dtype: np.dtype) -> bool:
    try:
        arr = np.load(path, mmap_mode='r')
        return arr.shape == (rows,) and arr.dtype == dtype
    except (OSError, ValueError):
        return False


def iteration_complete(paths: RunPaths, iteration: int) -> bool:
    summary = read_json(paths.summary(iteration))
    if not summary or summary.get('status') != 'complete':
        return False
    return valid_npy(paths.classification(iteration), summary['n_points'], CLASS_DTYPE)


def aggregate(paths: RunPaths, threads: int, log: logging.Logger) -> dict | None:
    """class probability for real objects = fraction of completed iterations
    assigned to that class. counting happens in c++."""
    done = []
    for summary_path in sorted((paths.root / 'summary').glob('iter_*.json')):
        iteration = int(summary_path.stem.split('_')[1])
        if iteration_complete(paths, iteration):
            done.append((iteration, read_json(summary_path)))
    if not done:
        log.info('aggregation skipped: no completed iterations')
        return None
    n_real = done[0][1]['n_real']
    if any(s['n_real'] != n_real for _, s in done):
        raise ValueError('completed iterations have different numbers of real objects')

    iterations = [it for it, _ in done]
    log.info('aggregating %s iterations: %s', len(iterations), iterations)
    counts = np.zeros((n_real, 4), dtype=np.uint16)
    reference_ids = None
    for iteration in iterations:
        cls = np.load(paths.classification(iteration), mmap_mode='r')
        ids = np.asarray(cls['TARGETID'][:n_real])
        if not np.all(cls['ISDATA'][:n_real]):
            raise ValueError(f'iter {iteration}: the first {n_real} rows are not all real objects')
        if reference_ids is None:
            reference_ids = ids
        elif not np.array_equal(ids, reference_ids):
            raise ValueError(f'iter {iteration}: real TARGETIDs differ between iterations')
        for start in range(0, n_real, CHUNK_ROWS):
            end = min(n_real, start + CHUNK_ROWS)
            _core.accumulate_webtypes(counts[start:end], np.ascontiguousarray(cls['WEBTYPE'][start:end]),
                                      threads=threads)
        del cls
    prob = _core.counts_to_probabilities(counts, len(iterations), threads=threads)
    columns = {'TARGETID': reference_ids, **{f: prob[:, k] for k, f in enumerate(PROB_FIELDS)}}
    write_structured(paths.probability(), columns, PROB_DTYPE)
    log.info('probabilities saved to %s', paths.probability())
    return {'iterations': iterations, 'path': str(paths.probability())}

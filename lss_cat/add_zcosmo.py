import argparse
import multiprocessing as mp
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fitsio
import numpy as np

MOCK_BASE = Path('/pscratch/sd/v/vtorresg/abacus_y5_iny5_only/SecondGenMocks/AbacusSummitBGS_v2')
LSS_DIR = MOCK_BASE / 'altmtl0/matterhorn-v2/mock0/LSScats'
FORFA = MOCK_BASE / 'forFA0.fits'
OUT_DIR = Path('/global/cfs/cdirs/desi/users/vtorresg/astra-mocks-v1/AbacusSummit_base_c000_ph000/'
               'BGS_z0.200/cutsky/fa_targets/LSScats')

# mainp.zmin/zmax for BGS in LSS/globals.py, applied to Z (with rsd) by mkclusdat
ZMIN, ZMAX = 0.002, 0.6

# table keywords that fitsio regenerates from the new dtype
STRUCTURAL = re.compile(r'^(XTENSION|BITPIX|NAXIS\d*|PCOUNT|GCOUNT|TFIELDS|EXTNAME|'
                        r'T(TYPE|FORM|UNIT|DIM|NULL|SCAL|ZERO|DISP)\d+)$')

# peak memory per worker in units of the input file size: input + output
# arrays + staged file in ram + lookups, with margin
MEM_PER_INPUT_BYTE = 4.0

# shared with the workers through fork (set in main before the pool starts)
_TRUEZ = None
_DATA_IDS = None


class TrueZ:
    """TARGETID -> (TRUEZ, RSDZ) lookup on forFA, sorted by TARGETID."""

    def __init__(self, path):
        self.name = path.name
        fa = fitsio.read(str(path), columns=['TARGETID', 'TRUEZ', 'RSDZ'])
        order = np.argsort(fa['TARGETID'], kind='stable')
        self.tid = fa['TARGETID'][order]
        self.truez = fa['TRUEZ'][order]
        self.rsdz = fa['RSDZ'][order]
        if np.any(np.diff(self.tid) <= 0):
            raise ValueError(f'{path.name}: TARGETID is not unique')
        print(f'forFA: {len(self.tid)} targets | {path}', flush=True)

    def index(self, ids, what):
        idx = np.searchsorted(self.tid, ids)
        np.clip(idx, 0, len(self.tid) - 1, out=idx)
        missing = np.count_nonzero(self.tid[idx] != ids)
        if missing:
            raise ValueError(f'{what}: {missing} ids not found in forFA')
        return idx


def log(name, msg):
    print(f'{time.strftime("%H:%M:%S")} [{name}] {msg}', flush=True)


def copy_header(hdr):
    """non-structural keywords of the input table header."""
    return [rec for rec in hdr.records() if not STRUCTURAL.match(rec['name'] or '')]


def with_z_cosmo(arr, z_cosmo):
    """copy of `arr` with Z_COSMO inserted right after Z."""
    names = arr.dtype.names
    if 'Z_COSMO' in names:
        raise ValueError('input already has a Z_COSMO column')
    descr = []
    for name in names:
        descr.append((name, arr.dtype[name]))
        if name == 'Z':
            descr.append(('Z_COSMO', arr.dtype['Z']))
    out = np.empty(len(arr), dtype=descr)
    for name in names:
        out[name] = arr[name]
    out['Z_COSMO'] = z_cosmo
    return out


def is_copy_of(out_path, in_path):
    """same columns, rows, TARGETID and Z as the input (a plain copy of it)."""
    with fitsio.FITS(str(out_path)) as fo, fitsio.FITS(str(in_path)) as fi:
        if fo[1].get_rec_dtype()[0] != fi[1].get_rec_dtype()[0] or fo[1].get_nrows() != fi[1].get_nrows():
            return False
        a = fo[1].read(columns=['TARGETID', 'Z'])
        b = fi[1].read(columns=['TARGETID', 'Z'])
    return np.array_equal(a, b)


def convert(in_path, out_path, id_col, overwrite, stage_dir):
    """add Z_COSMO to one catalog. returns a short status string."""
    truez, data_ids = _TRUEZ, _DATA_IDS if id_col == 'TARGETID_DATA' else None
    name = in_path.name
    t0 = time.monotonic()
    if out_path.resolve() == in_path.resolve():
        raise ValueError(f'refusing to overwrite the input {in_path}')
    if out_path.exists():
        with fitsio.FITS(str(out_path)) as f:
            done = 'Z_COSMO' in f[1].get_colnames()
        if done:
            if not overwrite:
                return f'skip (already has Z_COSMO, use --overwrite): {out_path}'
        elif is_copy_of(out_path, in_path):
            log(name, f'{out_path.name} is a copy of the input without Z_COSMO; it will be replaced')
        else:
            raise ValueError(f'{out_path} exists, has no Z_COSMO and differs from {in_path}; not replacing it')

    with fitsio.FITS(str(in_path)) as f:
        arr = f[1].read()
        hdr = f[1].read_header()
        extname = f[1].get_extname() or 'LSS'
    log(name, f'read {len(arr)} rows in {time.monotonic() - t0:.0f} s')

    idx = truez.index(arr[id_col], f'{name} [{id_col}]')
    if not np.array_equal(arr['Z'], truez.rsdz[idx]):
        bad = np.count_nonzero(arr['Z'] != truez.rsdz[idx])
        raise ValueError(f'{name}: Z != RSDZ for {bad} rows; not the expected catalog')
    if data_ids is not None:
        pos = np.clip(np.searchsorted(data_ids, arr[id_col]), 0, len(data_ids) - 1)
        orphan = np.count_nonzero(data_ids[pos] != arr[id_col])
        if orphan:
            raise ValueError(f'{name}: {orphan} TARGETID_DATA not in the data clustering catalog')

    z_cosmo = truez.truez[idx]
    z_rsd = arr['Z'].copy()
    out = with_z_cosmo(arr, z_cosmo)
    del arr, idx

    n_out = np.count_nonzero((z_cosmo <= ZMIN) | (z_cosmo >= ZMAX))
    dz = z_rsd.astype(np.float64) - z_cosmo
    log(name, f'rows={len(out)} | Z_COSMO in [{z_cosmo.min():.5f}, {z_cosmo.max():.5f}] | '
              f'outside ({ZMIN}, {ZMAX}): {n_out} | Z - Z_COSMO: mean={dz.mean():.2e} std={dz.std():.2e}')
    del dz

    header = copy_header(hdr) + [{'name': 'ZCOL', 'value': 'Z', 'comment': 'redshift with RSD (RSDZ)'},
                                 {'name': 'ZCOSMCOL', 'value': 'Z_COSMO', 'comment': f'redshift without RSD (TRUEZ, {truez.name})'},
                                 {'name': 'ZMATCH', 'value': id_col, 'comment': 'column matched to forFA TARGETID'},
                                 {'name': 'ZINPUT', 'value': name, 'comment': 'input clustering catalog'}]
    # write and verify in the staging directory (or next to the output), then
    # copy to a temporary name in the output directory and rename
    out_tmp = out_path.with_name(out_path.name + '.tmp')
    stage = Path(stage_dir) / f'{out_path.name}.{os.getpid()}.tmp' if stage_dir else out_tmp
    try:
        for p in {stage, out_tmp}:
            p.unlink(missing_ok=True)
        fitsio.write(str(stage), out, header=header, extname=extname, clobber=True)
        n_rows = len(out)
        del out

        with fitsio.FITS(str(stage)) as f:
            if f[1].get_nrows() != n_rows:
                raise IOError(f'{stage}: wrote {f[1].get_nrows()} rows, expected {n_rows}')
            check = f[1].read(columns=['Z', 'Z_COSMO'])
        if not (np.array_equal(check['Z'], z_rsd) and np.array_equal(check['Z_COSMO'], z_cosmo)):
            raise IOError(f'{stage}: Z / Z_COSMO read back do not match')
        del check
        log(name, f'written and verified in {time.monotonic() - t0:.0f} s')

        if stage != out_tmp:
            shutil.copyfile(stage, out_tmp)
            if out_tmp.stat().st_size != stage.stat().st_size:
                raise IOError(f'{out_tmp}: size differs from the staged file {stage}')
        os.replace(out_tmp, out_path)
    finally:
        for p in {stage, out_tmp}:
            p.unlink(missing_ok=True)
    return f'wrote {out_path} in {time.monotonic() - t0:.0f} s'


def random_indices(lss_dir, tracer):
    pattern = re.compile(rf'^{re.escape(tracer)}_(\d+)_clustering\.ran\.fits$')
    return sorted(int(m.group(1)) for p in lss_dir.iterdir() if (m := pattern.match(p.name)))


def mem_available():
    """MemAvailable from /proc/meminfo, in bytes (None if unknown)."""
    try:
        with open('/proc/meminfo') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def default_workers(tasks):
    cpus = len(os.sched_getaffinity(0))
    biggest = max(t[0].stat().st_size for t in tasks)
    avail = mem_available()
    by_mem = int(0.8 * avail / (MEM_PER_INPUT_BYTE * biggest)) if avail else 1
    return max(1, min(len(tasks), cpus, by_mem))


def main():
    global _TRUEZ, _DATA_IDS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lss-dir', type=Path, default=LSS_DIR, help='input LSScats directory (read only)')
    p.add_argument('--forfa', type=Path, default=FORFA, help='forFA file with TARGETID, TRUEZ, RSDZ')
    p.add_argument('--out-dir', type=Path, default=OUT_DIR, help='output directory')
    p.add_argument('--suffix', default='', help='added to the output names before .dat/.ran')
    p.add_argument('--tracer', default='BGS_ANY')
    p.add_argument('--ranmin', type=int, default=None, help='first random index (default: all found)')
    p.add_argument('--ranmax', type=int, default=None, help='last random index, exclusive')
    p.add_argument('--skip-data', action='store_true', help='only process the randoms')
    p.add_argument('--skip-randoms', action='store_true', help='only process the data')
    p.add_argument('--overwrite', action='store_true', help='rewrite outputs that already have Z_COSMO')
    p.add_argument('--workers', type=int, default=0,
                   help='files processed in parallel (0: automatic from available memory and cpus)')
    p.add_argument('--stage-dir', default='/dev/shm',
                   help="directory in ram to write and verify each file before copying it ('' to disable)")
    args = p.parse_args()

    lss_dir, out_dir = args.lss_dir.resolve(), args.out_dir.resolve()
    if lss_dir == out_dir:
        raise SystemExit('--out-dir must differ from --lss-dir')
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = args.stage_dir if args.stage_dir and os.access(args.stage_dir, os.W_OK) else None

    data_in = lss_dir / f'{args.tracer}_clustering.dat.fits'
    found = random_indices(lss_dir, args.tracer)
    ranmin = args.ranmin if args.ranmin is not None else (found[0] if found else 0)
    ranmax = args.ranmax if args.ranmax is not None else (found[-1] + 1 if found else 0)
    rans = [] if args.skip_randoms else list(range(ranmin, ranmax))
    missing = [n for n in rans if n not in found]
    if missing:
        raise SystemExit(f'random catalogs not found in {lss_dir}: {missing}')

    tasks = []
    if not args.skip_data:
        tasks.append((data_in, out_dir / f'{args.tracer}_clustering{args.suffix}.dat.fits', 'TARGETID'))
    for n in rans:
        tasks.append((lss_dir / f'{args.tracer}_{n}_clustering.ran.fits',
                      out_dir / f'{args.tracer}_{n}_clustering{args.suffix}.ran.fits', 'TARGETID_DATA'))
    if not tasks:
        raise SystemExit('nothing to do')
    workers = args.workers if args.workers > 0 else default_workers(tasks)
    print(f'input: {lss_dir}\noutput: {out_dir}\nrandoms: {rans}\n'
          f'workers: {workers} | stage: {stage_dir or "none (write directly)"}', flush=True)

    _TRUEZ = TrueZ(args.forfa)
    if rans:
        _DATA_IDS = np.sort(fitsio.read(str(data_in), columns=['TARGETID'])['TARGETID'])

    failed = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
        futures = {pool.submit(convert, i, o, c, args.overwrite, stage_dir): i.name for i, o, c in tasks}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                log(name, fut.result())
            except Exception as exc:
                failed.append(name)
                log(name, f'FAILED: {exc!r}')
    if failed:
        raise SystemExit(f'{len(failed)} files failed: {failed}')
    print('done', flush=True)


if __name__ == '__main__':
    main()
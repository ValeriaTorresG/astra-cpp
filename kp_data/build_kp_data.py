import argparse
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import fitsio
import numpy as np

KPDATA = 'v1'  # version of this layout, written as KPDATA in every file

ABACUS = Path('/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit')
CUBIC_FILE = ABACUS / 'CubicBox/BGS/v0/z0.200/AbacusSummit_base_c000_ph000/BGS_box_ph000.fits'
CUTSKY_DIR = ABACUS / 'CutSky/BGS/v0/z0.200'
CUTSKY_REAL = CUTSKY_DIR / 'cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits'
OUT_ROOT = Path('/global/cfs/cdirs/desicollab/users/vtorresg/astra-mocks-kp/data')
# clustering catalogs with Z and Z_COSMO (lss_cat/add_zcosmo.py)
LSS_DIR = OUT_ROOT / 'LSScats' / 'LSScats_full'
FORFA = Path('/pscratch/sd/v/vtorresg/abacus_y5_iny5_only/SecondGenMocks/AbacusSummitBGS_v2/forFA0.fits')

SIMNAME = 'AbacusSummit_base_c000_ph000'
ZSNAP = 0.2
BOX_SIZE = 2000.0
FOOTPRINT = 'IN_Y5'
TRACER = 'BGS_ANY'
# same as astra_cpp.config: random targetid = -(seed * factor + row + 1)
RANDOM_ID_FACTOR = 10_000_000_000
# mainp.zmin/zmax for BGS in LSS/globals.py, applied to Z (with rsd) by mkclusdat
LSS_ZMIN, LSS_ZMAX = 0.002, 0.6
MAG_COLS = ('R_MAG_APP', 'R_MAG_ABS', 'G_R_REST', 'G_R_OBS')
SAMPLES = {
    # sample: (catalog, rsd, altmtl, redshift column or None)
    'cubicbox': ('cubicbox', False, False, None),
    'cutsky_z': ('cutsky', True, False, 'Z'),
    'cutsky_z_cosmo': ('cutsky', False, False, 'Z_COSMO'),
    'LSScats_z': ('LSScats', True, True, 'Z'),
    'LSScats_z_cosmo': ('LSScats', False, True, 'Z_COSMO'),
}
# output directory of each sample, relative to the output root
SAMPLE_DIRS = {
    'cubicbox': 'cubicbox',
    'cutsky_z': 'cutsky/z',
    'cutsky_z_cosmo': 'cutsky/z_cosmo',
    'LSScats_z': 'LSScats/z',
    'LSScats_z_cosmo': 'LSScats/z_cosmo',
}

# table keywords that fitsio regenerates from the new dtype, and the ones from
# add_zcosmo.py that this script replaces
STRUCTURAL = re.compile(r'^(XTENSION|BITPIX|NAXIS\d*|PCOUNT|GCOUNT|TFIELDS|EXTNAME|'
                        r'T(TYPE|FORM|UNIT|DIM|NULL|SCAL|ZERO|DISP)\d+|ZCOL|ZCOSMCOL|ZMATCH|ZINPUT)$')

# shared with the workers through fork (set in main before the pool starts)
_FORFA = None


def log(name, msg):
    print(f'{time.strftime("%H:%M:%S")} [{name}] {msg}', flush=True)


class ForFA:
    """TARGETID -> magnitudes, TRUEZ and RSDZ on forFA, sorted by TARGETID."""

    def __init__(self, path):
        self.name = path.name
        fa = fitsio.read(str(path), columns=['TARGETID', 'TRUEZ', 'RSDZ', *MAG_COLS])
        fa = fa[np.argsort(fa['TARGETID'], kind='stable')]
        if np.any(np.diff(fa['TARGETID']) <= 0):
            raise ValueError(f'{path.name}: TARGETID is not unique')
        self.tid = fa['TARGETID']
        self.cols = fa
        print(f'forFA: {len(fa)} targets | {path}', flush=True)

    def rows(self, ids, what):
        idx = np.searchsorted(self.tid, ids)
        np.clip(idx, 0, len(self.tid) - 1, out=idx)
        missing = np.count_nonzero(self.tid[idx] != ids)
        if missing:
            raise ValueError(f'{what}: {missing} ids not found in forFA')
        return self.cols[idx]


@dataclass
class Task:
    """one source file and the samples written from it."""
    kind: str                       # cubicbox, cutsky or LSScats
    src: Path
    samples: list
    is_random: bool = False
    seed: int = 0                   # cutsky randoms: seed in the file name
    keys: list = field(default_factory=list)   # keywords specific to this source


def seed_of(path):
    return int(re.fullmatch(r'random_S(\d+)_\d+X\.fits', path.name).group(1))


def key(name, value, comment=''):
    return {'name': name, 'value': value, 'comment': comment}


def full_dtype(kind, src_dtype):
    """dtype with every column written by any sample of this kind."""
    descr = [(n, src_dtype[n]) for n in src_dtype.names]
    if kind == 'cutsky':
        return np.dtype([('TARGETID', '>i8')] + descr)
    if kind == 'LSScats':
        out = []
        for n, dt in descr:
            if n == 'R_MAG_ABS':
                out.append(('R_MAG_APP', dt))
            out.append((n, dt))
            if n == 'G_R_REST':
                out.append(('G_R_OBS', dt))
        return np.dtype(out)
    return np.dtype(descr)


def sample_dtype(full, zcol):
    """full dtype without the other redshift column."""
    drop = {'Z': 'Z_COSMO', 'Z_COSMO': 'Z'}.get(zcol)
    return np.dtype([(n, full[n]) for n in full.names if n != drop])


def expand(task, chunk, start, full):
    """rows of `chunk` (source rows start, start + 1, ...) kept in the outputs,
    with the added columns."""
    if task.kind == 'cubicbox':
        return chunk
    if task.kind == 'cutsky':
        rows = start + np.flatnonzero(chunk[FOOTPRINT] != 0)
        chunk = chunk[rows - start]
        out = np.empty(len(chunk), dtype=full)
        for n in chunk.dtype.names:
            out[n] = chunk[n]
        out['TARGETID'] = -(task.seed * RANDOM_ID_FACTOR + rows + 1) if task.is_random else rows + 1
        return out
    # LSScats: magnitudes from forFA, checked against the columns already present
    id_col = 'TARGETID_DATA' if task.is_random else 'TARGETID'
    fa = _FORFA.rows(chunk[id_col], f'{task.src.name} [{id_col}]')
    for col, fa_col in (('Z', 'RSDZ'), ('Z_COSMO', 'TRUEZ'), ('R_MAG_ABS', 'R_MAG_ABS'), ('G_R_REST', 'G_R_REST')):
        bad = np.count_nonzero(chunk[col] != fa[fa_col])
        if bad:
            raise ValueError(f'{task.src.name}: {col} != forFA {fa_col} for {bad} rows')
    out = np.empty(len(chunk), dtype=full)
    for n in chunk.dtype.names:
        out[n] = chunk[n]
    out['R_MAG_APP'] = fa['R_MAG_APP']
    out['G_R_OBS'] = fa['G_R_OBS']
    return out


def check_cols(task, zcol):
    if task.kind == 'cubicbox':
        return ['x', 'y', 'z']
    return ['TARGETID', zcol] + (['R_MAG_APP', 'G_R_OBS'] if task.kind == 'LSScats' else [])


def checksum(arr):
    """order-independent exact checksum of a numeric column (wraps modulo 2**64)."""
    a = np.ascontiguousarray(arr, dtype=arr.dtype.newbyteorder('='))
    return int(a.view(f'u{a.dtype.itemsize}').sum(dtype=np.uint64))


def is_done(path):
    if not path.exists():
        return False
    try:
        hdr = fitsio.read_header(str(path), 1)
        return hdr.get('KPDATA') == KPDATA and 'TESTROWS' not in hdr
    except OSError:
        return False


def header_for(task, sample, src_hdr, n_src, max_rows):
    catalog, rsd, altmtl, zcol = SAMPLES[sample]
    keys = [r for r in src_hdr.records() if not STRUCTURAL.match(r['name'] or '')]
    keys += [
        key('KPDATA', KPDATA, 'astra-mocks-kp data layout version'),
        key('CATALOG', catalog, 'cubicbox, cutsky or LSScats'),
        key('SAMPLE', sample, f'sample, in astra-mocks-kp/data/{SAMPLE_DIRS[sample]}'),
        key('RANDOM', task.is_random, 'T: random catalog, F: data (galaxies)'),
        key('RSD', rsd, 'T: redshift space (RSD), F: real space'),
        key('ALTMTL', altmtl, 'T: LSS clustering catalog after AltMTL'),
        key('ZCOL', zcol or 'NONE', {'Z': 'redshift with RSD', 'Z_COSMO': 'redshift without RSD',
                                     None: 'no redshift: x, y, z positions in Mpc/h'}[zcol]),
        key('SIMNAME', SIMNAME, 'simulation'),
        key('ZSNAP', ZSNAP, 'snapshot redshift'),
        key('TRACER', TRACER if catalog == 'LSScats' else 'BGS', 'tracer'),
    ]
    if catalog == 'cubicbox':
        keys += [key('BOXSIZE', BOX_SIZE, 'periodic box side (Mpc/h)'),
                 key('FOOTPRNT', 'BOX', 'no footprint cut')]
    elif catalog == 'cutsky':
        keys += [key('FOOTPRNT', 'Y5', f'rows with {FOOTPRINT} != 0'),
                 key('IDCOL', 'TARGETID', 'astra id: data row+1, randoms -(SEED*1e10+row+1)')]
    else:
        keys += [key('FOOTPRNT', 'Y5', 'AltMTL Y5 (DA3, matterhorn-v2, mock0)'),
                 key('ZCUTCOL', 'Z', 'LSS redshift cut applied on Z (with RSD)'),
                 key('ZMIN', LSS_ZMIN, 'LSS cut ZMIN < Z'),
                 key('ZMAX', LSS_ZMAX, 'LSS cut Z < ZMAX'),
                 key('MAGSRC', FORFA.name, 'R_MAG_APP, G_R_OBS added from this file'),
                 key('MAGMATCH', 'TARGETID_DATA' if task.is_random else 'TARGETID', 'matched to forFA TARGETID')]
    keys += task.keys
    keys += [key('SRCFILE', task.src.name, 'source catalog'),
             key('SRCDIR', str(task.src.parent), 'source directory'),
             key('SRCROWS', n_src, 'rows in the source catalog'),
             key('CREATOR', 'astra-cpp kp_data/build_kp_data.py'),
             key('DATE', datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S'), 'UTC')]
    if max_rows:
        keys.append(key('TESTROWS', max_rows, 'TEST: only the first rows of the source'))
    return keys


def build(task, out_root, overwrite, chunk_rows, max_rows):
    """writes the samples of one source file. returns a short status string."""
    t0 = time.monotonic()
    outs = {s: out_root / SAMPLE_DIRS[s] / task.src.name for s in task.samples}
    todo = [s for s, p in outs.items() if overwrite or not is_done(p)]
    if not todo:
        return 'skip (done, use --overwrite)'
    for s in todo:
        if outs[s].exists():
            log(task.src.name, f'{outs[s]} will be replaced')

    with fitsio.FITS(str(task.src)) as fin:
        hdu = fin[1]
        n_src = hdu.get_nrows()
        n_read = min(n_src, max_rows) if max_rows else n_src
        src_hdr = hdu.read_header()
        extname = hdu.get_extname() or 'CATALOG'
        full = full_dtype(task.kind, hdu.get_rec_dtype()[0])

        writers, stats = {}, {}
        try:
            for s in todo:
                zcol = SAMPLES[s][3]
                tmp = outs[s].with_name(outs[s].name + '.tmp')
                tmp.parent.mkdir(parents=True, exist_ok=True)
                f = fitsio.FITS(str(tmp), 'rw', clobber=True)
                f.create_table_hdu(dtype=sample_dtype(full, zcol), extname=extname)
                f[-1].write_keys(header_for(task, s, src_hdr, n_src, max_rows))
                writers[s] = (f, tmp)
                stats[s] = {'rows': 0, 'sums': dict.fromkeys(check_cols(task, zcol), 0)}

            for start in range(0, n_read, chunk_rows):
                stop = min(start + chunk_rows, n_read)
                rows = expand(task, hdu[start:stop], start, full)
                for s in todo:
                    f, _ = writers[s]
                    names = f[-1].get_rec_dtype()[0].names
                    out = np.empty(len(rows), dtype=sample_dtype(full, SAMPLES[s][3]))
                    for n in names:
                        out[n] = rows[n]
                    if len(out):
                        f[-1].append(out)
                    st = stats[s]
                    st['rows'] += len(out)
                    for c in st['sums']:
                        st['sums'][c] += checksum(out[c])
                if (start // chunk_rows) % 10 == 0:
                    log(task.src.name, f'{stop}/{n_read} rows ({time.monotonic() - t0:.0f} s)')
            for f, _ in writers.values():
                f.close()

            # read back and verify before renaming
            for s in todo:
                _, tmp = writers[s]
                st = stats[s]
                sums = dict.fromkeys(st['sums'], 0)
                with fitsio.FITS(str(tmp)) as f:
                    if f[1].get_nrows() != st['rows']:
                        raise IOError(f'{tmp}: wrote {f[1].get_nrows()} rows, expected {st["rows"]}')
                    for b in range(0, st['rows'], chunk_rows):
                        part = f[1][list(sums)][b:b + chunk_rows]
                        for c in sums:
                            sums[c] += checksum(part[c])
                for c in sums:
                    sums[c] %= 2**64
                    st['sums'][c] %= 2**64
                if sums != st['sums']:
                    raise IOError(f'{tmp}: columns read back do not match what was written')
                os.replace(tmp, outs[s])
                log(task.src.name, f'{s}: {st["rows"]} of {n_read} rows -> {outs[s]}')
        finally:
            for f, tmp in writers.values():
                f.close()
                tmp.unlink(missing_ok=True)
    return f'done in {time.monotonic() - t0:.0f} s'


def make_tasks(only, max_randoms):
    tasks = []
    if 'cubicbox' in only:
        tasks.append(Task('cubicbox', CUBIC_FILE, ['cubicbox']))
    if 'cutsky' in only:
        samples = ['cutsky_z', 'cutsky_z_cosmo']
        tasks.append(Task('cutsky', CUTSKY_REAL, samples))
        randoms = sorted(CUTSKY_DIR.glob('random_S*_1X.fits'), key=seed_of)[:max_randoms]
        for p in randoms:
            tasks.append(Task('cutsky', p, samples, is_random=True, seed=seed_of(p),
                              keys=[key('SEED', seed_of(p), 'random seed')]))
    if 'lss' in only:
        samples = ['LSScats_z', 'LSScats_z_cosmo']
        tasks.append(Task('LSScats', LSS_DIR / f'{TRACER}_clustering.dat.fits', samples))
        pattern = re.compile(rf'^{TRACER}_(\d+)_clustering\.ran\.fits$')
        found = sorted(int(m.group(1)) for p in LSS_DIR.iterdir() if (m := pattern.match(p.name)))
        for n in found[:max_randoms]:
            tasks.append(Task('LSScats', LSS_DIR / f'{TRACER}_{n}_clustering.ran.fits', samples, is_random=True,
                              keys=[key('RANIDX', n, 'LSS random index')]))
    missing = [str(t.src) for t in tasks if not t.src.exists()]
    if missing:
        raise SystemExit(f'source catalogs not found: {missing}')
    return tasks


README_HEAD = """\
# astra-mocks-kp input catalogs

Input catalogs for the ASTRA cosmic-web classifications of the AbacusSummit
`base_c000_ph000` BGS mocks at z = 0.200. The `z` and `z_cosmo` directories
hold the same objects; each file has a single redshift column (`Z` with RSD or
`Z_COSMO` without RSD), so the redshift space is selected by directory.
Written by `kp_data/build_kp_data.py` in `astra-cpp`.

```
data/
├── cubicbox/              cubic box, 2000 Mpc/h periodic, real space
├── cutsky/                CutSky galaxies and 30 randoms, DESI Y5 footprint (IN_Y5 != 0)
│   ├── z/                 Z (with RSD)
│   └── z_cosmo/           Z_COSMO (without RSD)
└── LSScats/               BGS_ANY clustering catalogs, DESI Y5 footprint, after AltMTL fiber assignment (data and 18 randoms)
    ├── z/                 Z (with RSD), plus R_MAG_APP, R_MAG_ABS, G_R_REST, G_R_OBS
    ├── z_cosmo/           Z_COSMO (without RSD), same columns
    └── LSScats_full/      input of z/ and z_cosmo/: both Z and Z_COSMO, no R_MAG_APP or G_R_OBS
```

| Directory | `SAMPLE` | RSD | AltMTL |
|---|---|---|---|
| `cubicbox/` | `cubicbox` | F | F |
| `cutsky/z/` | `cutsky_z` | T | F |
| `cutsky/z_cosmo/` | `cutsky_z_cosmo` | F | F |
| `LSScats/z/` | `LSScats_z` | T | T |
| `LSScats/z_cosmo/` | `LSScats_z_cosmo` | F | T |

Sources: `{cubic}`, `{cutsky_dir}/` (`{cutsky_real}` and `random_S*_1X.fits`) and
`{lss_dir}/` (LSS `mkCat_SecondGen_amtl.py --survey=DA3 --specdata=matterhorn-v2
--mocknum=0`, with `Z_COSMO` added by `lss_cat/add_zcosmo.py`).

## Header keywords (HDU 1)

| Keyword | Meaning |
|---|---|
| `KPDATA` | layout version (`{version}`) |
| `CATALOG` | `cubicbox`, `cutsky` or `LSScats` |
| `SAMPLE` | sample name (table above) |
| `RANDOM` | T: random catalog, F: data (galaxies) |
| `RSD` | T: redshift space, F: real space |
| `ALTMTL` | T: LSS clustering catalog after AltMTL fiber assignment |
| `ZCOL` | redshift column in the file (`Z`, `Z_COSMO`, or `NONE` for the box) |
| `FOOTPRNT` | `Y5` or `BOX` |
| `SIMNAME`, `ZSNAP`, `TRACER` | simulation, snapshot redshift, tracer |
| `BOXSIZE` | box side in Mpc/h (cubicbox) |
| `SEED` | random seed (cutsky randoms, from `random_S{{SEED}}_1X.fits`) |
| `RANIDX` | random index (LSScats randoms) |
| `IDCOL` | cutsky: `TARGETID` follows the ASTRA convention (see below) |
| `ZCUTCOL`, `ZMIN`, `ZMAX` | LSScats: redshift cut of the LSS pipeline |
| `MAGSRC`, `MAGMATCH` | LSScats: file and column used to add the magnitudes |
| `SRCFILE`, `SRCDIR`, `SRCROWS` | source catalog and its number of rows |

```python
import fitsio
hdr = fitsio.read_header(path, 1)
hdr['RSD'], hdr['ALTMTL'], hdr['RANDOM'], hdr['ZCOL']
```

## Notes

- **cutsky**: all CutSky columns are kept except the other redshift. A
  `TARGETID` column is added with the ids used by ASTRA on the full CutSky
  files: data = one-based row in the CutSky file, randoms =
  `-(SEED * 10**10 + row + 1)` with `row` the zero-based row in
  `random_S{{SEED}}_1X.fits`. They match the `TARGETID` of the ASTRA outputs in
  `AbacusSummit_base_c000_ph000/BGS_z0.200/cutsky/all_targets/`. ASTRA run
  directly on these files numbers the rows of the filtered files instead.
- **LSScats**: the LSS pipeline selects `{zmin} < Z < {zmax}` with `Z` (with RSD),
  so `LSScats/z_cosmo` has the same galaxies and some `Z_COSMO` values fall
  outside that range. `R_MAG_APP` and `G_R_OBS` come from `{forfa}`, matched
  by `TARGETID` (data) or `TARGETID_DATA` (randoms, which inherit them from
  their source galaxy, like `Z`); `R_MAG_ABS`, `G_R_REST`, `Z` and `Z_COSMO`
  were checked to be equal to the forFA values (`RSDZ`, `TRUEZ`). Files
  written before the move to `LSScats/LSScats_full/` have
  `SRCDIR = .../astra-mocks-kp/data/LSScats`; the source is the file with the
  same name in `LSScats/LSScats_full/`.
- **cubicbox**: same columns as the AbacusSummit box (no redshift column).

## Files

| File | Rows | RANDOM | RSD | ALTMTL | ZCOL |
|---|---|---|---|---|---|
"""


def write_readme(out_root):
    lines = []
    for sample in SAMPLES:
        for path in sorted((out_root / SAMPLE_DIRS[sample]).glob('*.fits')):
            hdr = fitsio.read_header(str(path), 1)
            if hdr.get('KPDATA') != KPDATA:
                continue
            flag = lambda k: 'T' if hdr[k] else 'F'
            lines.append(f'| `{SAMPLE_DIRS[sample]}/{path.name}` | {hdr["NAXIS2"]:,} | {flag("RANDOM")} | '
                         f'{flag("RSD")} | {flag("ALTMTL")} | `{hdr["ZCOL"]}` |')
    text = README_HEAD.format(version=KPDATA, cubic=CUBIC_FILE, cutsky_dir=CUTSKY_DIR,
                              cutsky_real=CUTSKY_REAL.name, lss_dir=LSS_DIR, zmin=LSS_ZMIN,
                              zmax=LSS_ZMAX, forfa=FORFA)
    (out_root / 'README.md').write_text(text + '\n'.join(lines) + '\n')
    print(f'README: {out_root / "README.md"} ({len(lines)} files)', flush=True)


def main():
    global _FORFA
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', type=Path, default=OUT_ROOT, help='output root (one subdirectory per sample)')
    p.add_argument('--only', nargs='+', choices=('cubicbox', 'cutsky', 'lss'), default=['cubicbox', 'cutsky', 'lss'])
    p.add_argument('--overwrite', action='store_true', help='rewrite outputs that are already done')
    p.add_argument('--workers', type=int, default=16, help='source files processed in parallel')
    p.add_argument('--chunk-rows', type=int, default=4_000_000, help='rows read per chunk')
    p.add_argument('--max-rows', type=int, default=None, help='TEST: only the first rows of each source')
    p.add_argument('--max-randoms', type=int, default=None, help='TEST: only the first random files')
    p.add_argument('--readme-only', action='store_true', help='only rewrite README.md from the existing outputs')
    args = p.parse_args()

    out_root = args.out_dir.resolve()
    if not args.readme_only:
        tasks = make_tasks(args.only, args.max_randoms)
        workers = max(1, min(args.workers, len(tasks)))
        print(f'output: {out_root}\nsources: {len(tasks)} | workers: {workers}', flush=True)
        if 'lss' in args.only:
            _FORFA = ForFA(FORFA)

        failed = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
            futures = {pool.submit(build, t, out_root, args.overwrite, args.chunk_rows, args.max_rows): t.src.name
                       for t in tasks}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    log(name, fut.result())
                except Exception as exc:
                    failed.append(name)
                    log(name, f'FAILED: {exc!r}')
        if failed:
            raise SystemExit(f'{len(failed)} files failed: {failed}')
    write_readme(out_root)
    print('done', flush=True)


if __name__ == '__main__':
    main()
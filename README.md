# astra-cpp

This version of ASTRA runs in c++: fits reading, cosmology, random pooling,
3d delaunay triangulation with CGAL (sequential or parallel with tbb), ndata/nrand
neighbor counts, r, and classification. Python orchestrates the stages and saves
the outputs. It runs on AbacusSummit BGS z0.2 mocks for DESI.


## Build

```bash
source /pscratch/sd/v/vtorresg/venvs/triang/bin/activate
bash build.sh              # creates python/astra_cpp/_core*.so
python -m unittest discover -s tests -v
```

Dependencies: cgal 5.6.2 (headers at `/pscratch/sd/v/vtorresg/deps/CGAL-5.6.2`),
gmp/mpfr from nersc python, system boost 1.66 and tbb 2019, openmp, and pybind11
from the virtual environment. set `CGAL_DIR` and `NERSC_PY` to change the paths;
`ASTRA_WITH_TBB=OFF` builds without parallel triangulation.


## Run one iteration

```bash
sbatch jobs/run_cutsky_iteration.sbatch                  # iteration 0, full cpu node
ITERATION=3 sbatch jobs/run_cutsky_iteration.sbatch
sbatch --array=0-9 jobs/run_cutsky_iteration.sbatch      # 10 iterations
sbatch jobs/run_cubic_iteration.sbatch                   # full cubic box, iteration 0

# manually, from the repository root
export PYTHONPATH=$PWD/python
python -m astra_cpp cutsky --iteration 0 --threads 128
python -m astra_cpp cutsky --iteration 0 --max-real-rows 1000000 --threads 32   # test
python -m astra_cpp cubic --iteration 0 --threads 128                           # full box
python -m astra_cpp cubic --iteration 0 --subbox 500                            # test (nonperiodic)
python -m astra_cpp aggregate /pscratch/sd/v/vtorresg/astra-cpp-runs/cutsky/<catalog>/z
```

By default, cut-sky runs keep only the real and random rows with `IN_Y5 != 0`
(DESI Y5 footprint); `--footprint COLUMN` picks another flag and `--no-footprint`
uses the full catalogs.


## Performance (full iteration, one cpu node)

The full box needs an entire cpu node (~333 gib out of ~460 available). Memory
is estimated before triangulation, with a warning if the run is unlikely to fit.

CGAL parallel insertion scales poorly beyond ~64 threads: lock contention becomes
erratic (40m points: 11.5 s with 64 threads, 30.9 s with 96), and a coarse grid
caused livelock. Triangulation therefore uses `min(--threads, 64)` threads by
default (`--triangulation-threads` overrides this) and a lock grid proportional
to n^(1/3) (`--lock-grid-cells`).

## Void finder

`astra_cpp.voids` implements the astra void finder ([forero/ASTRA-DESI, branch
void-finder](https://github.com/forero/ASTRA-DESI/tree/void-finder), `group_finder/`)
with the same c++ core. It is a separate command, but it reuses the classification
pieces: the same catalog loaders and seeds (so an iteration uses exactly the same
data + randoms as the classification), the cgal triangulation, and the neighbor counts.

1. r = (ndata - nrand) / (ndata + nrand) for every point; points with
   r <= `--r-threshold` (-0.25) are processed by ascending r (ties by index);
2. watershed: a point without an assigned neighbor seeds a new group, otherwise
   it joins the lowest-id neighboring group (basins never merge); groups with fewer
   than `--min-members` (4) data + randoms are removed;
3. cut-sky: angular (healpix ring, `--healpix-nside` 128, >= 3 randoms per pixel) and
   radial (5 mpc bins, >= 3 randoms) selection counted from the randoms of the
   iteration. members in invalid regions and internal edges crossing them are
   removed, each group keeps its seed's connected component, and affected groups
   are flagged `BORDER`;
4. shape from the random members only: center = mean position, moment tensor
   eigenvalues l1 >= l2 >= l3, `R_EFF = sqrt(5) (l1 l2 l3)^(1/6)`,
   `ELLIP = 1 - ((l3 + l2) / (l2 + l1))^(1/4)`. catalog coordinates are in mpc/h
   (cut-sky positions are multiplied by the planck18 h, `--h` overrides it);
5. `consensus`: pool the clean voids of several iterations, group them
   largest-first by contained volume fraction (> 0.5), take each group's
   lower-median-radius vote, prune overlapping representatives, and keep those
   found in more than half of the iterations.

Everything after the triangulation runs in c++: a stable parallel counting sort
by r (r takes only the values of the (ndata, degree) pairs), the subgraph induced
by the selected points stored once per edge in processing order (the watershed
streams it), a lock-free union-find for the mask pruning, and a grid for the
consensus queries. On top of the classification these stages add a few seconds and
~25 gb for the full cubic box.

```bash
python -m astra_cpp.voids cutsky --iteration 0 --threads 128     # same options as `astra_cpp cutsky`
python -m astra_cpp.voids cubic --iteration 0 --threads 128      # periodic box
python -m astra_cpp.voids cubic --iteration 0 --subbox 500       # test (nonperiodic, border by face buffer)
python -m astra_cpp.voids consensus /pscratch/sd/v/vtorresg/astra-cpp-runs/cutsky/<catalog>/z
sbatch jobs/run_cutsky_voids.sbatch                              # or jobs/run_cubic_voids.sbatch
```

Outputs go to `<run dir>/voids/` (the classification run directory):

    catalog/iter_XXX.npy     VOID_ID GROUP_ID XCART YCART ZCART R_EFF ELLIP EIGVAL_1..3 EIGVEC_i_X..Z
                             N_MEMBERS N_DATA N_RANDOM BORDER BORDER_FLAGS   (all groups after the mask)
    clean/iter_XXX.npy       BORDER == False rows, without the border columns
    membership/iter_XXX.npy  TARGETID ISDATA R SELECTED GROUP_ID_PREMASK GROUP_ID VOID_ID PRUNED BORDER,
                             one row per point, aligned with points/iter_XXX.npy (written if missing)
    summary/iter_XXX.json, logs/iter_XXX.log
    consensus/voids_nNNN.npy VOID_ID SRC_ITER X Y Z R_EFF ELLIP EIGVAL/EIGVEC V FRAC_V N_ABSORBED

`VOID_ID = iteration * 10^10 + GROUP_ID`, and `GROUP_ID` is the seed number in
processing order. `BORDER_FLAGS` bits are `_core.BORDER_*` (angular/radial member
or edge, box face). Differences from astra-desi:

- outputs are `.npy`, and the membership covers data and randoms;
- periodic boxes (the default for the cubic box) have no border: members are
  unwrapped around one of them, centers are wrapped back into the box, the
  consensus uses minimum-image distances, and groups spanning more than half the
  box are counted in `n_wrap_ambiguous`; nonperiodic boxes flag groups within the
  mean random spacing of a face (`--boundary-buffer`), as `run_quijote_voids.py`;
- exact duplicate points (no delaunay vertex of their own) take the group of their
  canonical point (astra-desi rejects duplicates);
- consensus pruning breaks exact radius ties by the lowest representative index
  (astra-desi depends on the kd-tree order) and keeps a representative once if two
  groups pick it (astra-desi raises).

`python -m unittest discover -s tests -p "test_voids*.py"` compares every stage with
a numpy/scipy port of astra-desi's group_finder (watershed, mask pruning, shapes,
consensus) on synthetic catalogs with an angular hole and a radial gap.

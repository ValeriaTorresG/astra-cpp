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


## Performance (full iteration, one cpu node)

The full box needs an entire cpu node (~333 gib out of ~460 available). Memory
is estimated before triangulation, with a warning if the run is unlikely to fit.

CGAL parallel insertion scales poorly beyond ~64 threads: lock contention becomes
erratic (40m points: 11.5 s with 64 threads, 30.9 s with 96), and a coarse grid
caused livelock. Triangulation therefore uses `min(--threads, 64)` threads by
default (`--triangulation-threads` overrides this) and a lock grid proportional
to n^(1/3) (`--lock-grid-cells`).
#!/bin/bash
# one astra iteration (c++/cgal core) on the abacussummit bgs cubic box.
# the full box has 232m galaxies + 232m uniform randoms, plus ~6% periodic
# copies (~493m points): ~340 gb peak, requiring a full cpu node.
#   sbatch jobs/run_cubic_iteration.sbatch                # full box, iteration 0
#   ITERATION=1 sbatch jobs/run_cubic_iteration.sbatch
#   sbatch jobs/run_cubic_iteration.sbatch --subbox 500   # test (nonperiodic subbox)
#SBATCH --job-name=astra_cpp_cubic
#SBATCH --account=desi
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --output=/pscratch/sd/v/vtorresg/astra-cpp-runs/slurm/%x_%j.out
#SBATCH --chdir=/global/homes/v/vtorresg/astra-cpp

set -euo pipefail

source /pscratch/sd/v/vtorresg/venvs/triang/bin/activate
export PYTHONPATH=$PWD/python${PYTHONPATH:+:$PYTHONPATH}

ITERATION=${ITERATION:-${SLURM_ARRAY_TASK_ID:-0}}
THREADS=${THREADS:-128}

python -m astra_cpp info
srun --cpu-bind=none python -m astra_cpp cubic \
  --iteration "$ITERATION" \
  --threads "$THREADS" \
  --output-dir /pscratch/sd/v/vtorresg/astra-cpp-runs \
  "$@"

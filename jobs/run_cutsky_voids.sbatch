#!/bin/bash
# astra void finder (c++/cgal core) for one iteration of the abacussummit bgs
# cut-sky mock. same catalog and seed as run_cutsky_iteration.sbatch; outputs in
# <run dir>/voids/ (membership rows aligned with <run dir>/points/iter_XXX.npy).
#   sbatch jobs/run_cutsky_voids.sbatch                    # iteration 0
#   ITERATION=3 sbatch jobs/run_cutsky_voids.sbatch
#   sbatch --array=0-9%2 jobs/run_cutsky_voids.sbatch      # iteration = SLURM_ARRAY_TASK_ID
#   sbatch jobs/run_cutsky_voids.sbatch --max-real-rows 1000000   # test
# consensus over the completed iterations (seconds, can run on a login node):
#   python -m astra_cpp.voids consensus /pscratch/sd/v/vtorresg/astra-cpp-runs/cutsky/<catalog>/z
#SBATCH --job-name=astra_cpp_cutsky_voids
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
THREADS=${THREADS:-128}          # one thread per physical core
REDSHIFT=${REDSHIFT:-Z}

python -m astra_cpp info
srun --cpu-bind=none python -m astra_cpp.voids cutsky \
  --iteration "$ITERATION" \
  --redshift "$REDSHIFT" \
  --threads "$THREADS" \
  --output-dir /pscratch/sd/v/vtorresg/astra-cpp-runs \
  "$@"

#!/bin/bash
# astra void finder (c++/cgal core) for one iteration of the abacussummit bgs
# cubic box (periodic by default, same randoms as run_cubic_iteration.sbatch).
# the triangulation dominates the memory as in the classification (~340 gb peak),
# plus ~25 gb for the positions, processing order and induced graph: full cpu node.
#   sbatch jobs/run_cubic_voids.sbatch                     # full box, iteration 0
#   ITERATION=1 sbatch jobs/run_cubic_voids.sbatch
#   sbatch jobs/run_cubic_voids.sbatch --subbox 500        # test (nonperiodic subbox)
#   sbatch jobs/run_cubic_voids.sbatch --no-membership     # skip the ~15 gb membership file
# consensus over the completed iterations:
#   python -m astra_cpp.voids consensus /pscratch/sd/v/vtorresg/astra-cpp-runs/cubic/BGS_box_ph000
#SBATCH --job-name=astra_cpp_cubic_voids
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
srun --cpu-bind=none python -m astra_cpp.voids cubic \
  --iteration "$ITERATION" \
  --threads "$THREADS" \
  --output-dir /pscratch/sd/v/vtorresg/astra-cpp-runs \
  "$@"

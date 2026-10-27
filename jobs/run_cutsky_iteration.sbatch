#!/bin/bash
# Una iteración de ASTRA (núcleo C++/CGAL) sobre el mock cut-sky BGS de AbacusSummit.
# Uso:
#   sbatch jobs/run_cutsky_iteration.sbatch                 # iteración 0
#   ITERATION=3 sbatch jobs/run_cutsky_iteration.sbatch
#   sbatch --array=0-9%2 jobs/run_cutsky_iteration.sbatch   # iteración = SLURM_ARRAY_TASK_ID
#SBATCH --job-name=astra_cpp_cutsky
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
THREADS=${THREADS:-128}          # un hilo por núcleo físico
REDSHIFT=${REDSHIFT:-Z}

python -m astra_cpp info
srun --cpu-bind=none python -m astra_cpp cutsky \
  --iteration "$ITERATION" \
  --redshift "$REDSHIFT" \
  --threads "$THREADS" \
  --output-dir /pscratch/sd/v/vtorresg/astra-cpp-runs \
  "$@"

#!/bin/bash
# Una iteración de ASTRA (núcleo C++/CGAL) sobre la caja cúbica BGS de AbacusSummit.
# La caja completa son 232M galaxias + 232M randoms uniformes, más ~6 % de copias
# periódicas (~493M puntos): pico de ~340 GB, necesita un nodo CPU completo.
#   sbatch jobs/run_cubic_iteration.sbatch                # caja completa, iteración 0
#   ITERATION=1 sbatch jobs/run_cubic_iteration.sbatch
#   sbatch jobs/run_cubic_iteration.sbatch --subbox 500   # prueba (sub-caja no periódica)
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

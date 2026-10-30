#!/bin/bash
#SBATCH --job-name=grid-search
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/grid-search-%A_%a.out
#SBATCH --error=logs/grid-search-%A_%a.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from the repo root}"  # run the checkout this was submitted from
cd "$REPO_ROOT"
mkdir -p logs results/tuning_provenance/grid_search_results

PYBIN=${DATECT_PY:-/gscratch/stf/ac283/envs/datect_scratch/bin/python}
export PATH="$(dirname "$PYBIN"):$PATH"

echo "== Node: $(hostname) =="
echo "== Array task: $SLURM_ARRAY_TASK_ID =="
echo "== Started: $(date) =="

$PYBIN scripts/eval/grid_search_weights_clip.py

echo "== Done: $(date) =="

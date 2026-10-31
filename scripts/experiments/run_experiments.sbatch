#!/bin/bash
# Forecast experiments (docs/FORECAST_EXPERIMENTS.md). Submit from the repo root of a
# disposable worktree:
#   WINDOWS="val dev" VARIANTS=all EXP_OUT=/gscratch/scrubbed/$USER/experiments/raw \
#     sbatch scripts/experiments/run_experiments.sbatch
# Re-queued (preempted) runs skip variants whose output already exists.
#SBATCH --job-name=fx-exp
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=logs/fx_exp_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=${DATECT_PY:-/gscratch/stf/ac283/envs/datect_scratch/bin/python}
OUT=${EXP_OUT:-/gscratch/scrubbed/$USER/experiments/raw}
for w in ${WINDOWS:-val}; do
  "$PY" -u scripts/experiments/harness.py --window "$w" --variants "${VARIANTS:-all}" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-16}" --out-dir "$OUT" --skip-existing --verify
done
echo "done $(date)"

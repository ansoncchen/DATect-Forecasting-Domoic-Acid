#!/bin/bash
# Pooled single-model comparison (docs/FORECAST_EXPERIMENTS.md §3b). From a worktree root:
#   WINDOWS="val" EXP_OUT=/gscratch/scrubbed/$USER/experiments/raw \
#     sbatch scripts/experiments/run_pooled.sbatch
# Feature-group ablation (diagnostic): add ABLATION=F1:linear (feature set:family[,family]).
#SBATCH --job-name=fx-pooled
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=logs/fx_pooled_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=${DATECT_PY:-/gscratch/stf/ac283/envs/datect_scratch/bin/python}
OUT=${EXP_OUT:-/gscratch/scrubbed/$USER/experiments/raw}
for w in ${WINDOWS:-val}; do
  "$PY" -u scripts/experiments/pooled_study.py --window "$w" --n-jobs "${SLURM_CPUS_PER_TASK:-16}" --out-dir "$OUT" \
    ${ABLATION:+--ablation "$ABLATION"}
done
echo "done $(date)"

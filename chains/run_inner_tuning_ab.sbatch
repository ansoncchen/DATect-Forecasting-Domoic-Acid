#!/bin/bash
#SBATCH --job-name=inner-tune-ab
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/inner-tune-ab-%A_%a.out
#SBATCH --error=logs/inner-tune-ab-%A_%a.err

set -euo pipefail
REPO_ROOT="/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid"
cd "$REPO_ROOT"
mkdir -p logs chains/output

PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH

# Task 0 = inner tuning ENABLED (default behavior). Don't set the env var.
# Task 1 = inner tuning DISABLED.
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    LABEL="inner_tuning_enabled"
    echo "== Task 0: inner tuning ENABLED (default DATect) =="
elif [ "$SLURM_ARRAY_TASK_ID" = "1" ]; then
    LABEL="inner_tuning_disabled"
    export DATECT_MIN_TRAINING_FOR_TUNING=99999
    echo "== Task 1: inner tuning DISABLED (DATECT_MIN_TRAINING_FOR_TUNING=99999) =="
fi

echo "== Node: $(hostname) =="
echo "== Started: $(date) =="

# Run the standard retrospective via the established subprocess pattern.
# Use validate_tuned_on_holdout.py with --label so we get a parquet of
# per-test-point predictions for both windows analysis.
$PYBIN scripts/eval/validate_tuned_on_holdout.py --label "$LABEL"

echo "== Done: $(date) =="

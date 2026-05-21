#!/bin/bash
#SBATCH --job-name=oad-holdout
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/oad-holdout-%A_%a.out
#SBATCH --error=logs/oad-holdout-%A_%a.err

set -euo pipefail
REPO_ROOT="/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid"
cd "$REPO_ROOT"
mkdir -p logs holdout_validation
PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH
export DATECT_MIN_TRAINING_FOR_TUNING=99999

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    echo "== baseline (current per_site_models.py) =="
    $PYBIN scripts/eval/validate_tuned_on_holdout.py --label baseline
elif [ "$SLURM_ARRAY_TASK_ID" = "1" ]; then
    echo "== tuned (proposed_overrides.json) =="
    DATECT_HPARAM_OVERRIDE_JSON="$REPO_ROOT/proposed_overrides.json" \
        $PYBIN scripts/eval/validate_tuned_on_holdout.py --label tuned
fi
echo "== done $(date) =="

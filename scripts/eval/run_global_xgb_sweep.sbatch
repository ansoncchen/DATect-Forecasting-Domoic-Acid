#!/bin/bash
#SBATCH --job-name=xgb-sweep
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/xgb-sweep-%A_%a.out
#SBATCH --error=logs/xgb-sweep-%A_%a.err

set -euo pipefail
cd /gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid
mkdir -p logs xgb_sweep_results

PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH
export DATECT_MIN_TRAINING_FOR_TUNING=99999

# 2x2x2 grid: max_depth × n_estimators × learning_rate
DEPTHS=(4 6 4 6 4 6 4 6)
NESTS=(400 400 600 600 400 400 600 600)
LRS=(0.05 0.05 0.05 0.05 0.10 0.10 0.10 0.10)

DEPTH=${DEPTHS[$SLURM_ARRAY_TASK_ID]}
NEST=${NESTS[$SLURM_ARRAY_TASK_ID]}
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

# Write per-task XGB override JSON for all 10 sites — overrides xgb_params, leaves
# ensemble_weights / clip / feature_subset alone (those are grid-winner from per_site_models.py).
TMPJSON=/tmp/xgb_sweep_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json
$PYBIN -c "
import json, sys; sys.path.insert(0, '.')
from forecasting.per_site_models import SITE_SPECIFIC_CONFIGS
o = {}
for site in SITE_SPECIFIC_CONFIGS:
    o[site] = {'xgb_params': {'max_depth': $DEPTH, 'n_estimators': $NEST, 'learning_rate': $LR}}
json.dump(o, open('$TMPJSON','w'))
"
export DATECT_HPARAM_OVERRIDE_JSON=$TMPJSON

echo "== Started: $(date) — depth=$DEPTH n_est=$NEST lr=$LR =="
$PYBIN scripts/eval/multi_seed_baseline.py --seed 42 --out-dir xgb_sweep_results/d${DEPTH}_n${NEST}_lr${LR}
echo "== Done: $(date) =="

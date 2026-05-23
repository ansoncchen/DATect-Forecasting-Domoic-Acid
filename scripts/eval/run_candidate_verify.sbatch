#!/bin/bash
#SBATCH --job-name=cand-verify
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/cand-verify-%A_%a.out
#SBATCH --error=logs/cand-verify-%A_%a.err

set -euo pipefail
cd /gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid
mkdir -p logs candidate_verify_results
PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH
export DATECT_MIN_TRAINING_FOR_TUNING=99999

# 0-4 = candidate A (mcw_x0.67 perturbation, seeds 42-46)
# 5-9 = candidate B (smaller spike classifier, seeds 42-46)
TASK=$SLURM_ARRAY_TASK_ID
SEED=$((42 + (TASK % 5)))

if [ $TASK -lt 5 ]; then
    # Candidate A: mcw × 0.67 on all per-site xgb_params
    TMPJSON=/tmp/cand_a_${SLURM_ARRAY_JOB_ID}_${TASK}.json
    $PYBIN -c "
import json, sys; sys.path.insert(0, '.')
from forecasting.per_site_models import SITE_SPECIFIC_CONFIGS
o = {}
for site, cfg in SITE_SPECIFIC_CONFIGS.items():
    if cfg.get('xgb_params') is None: continue
    xp = dict(cfg['xgb_params'])
    xp['min_child_weight'] = max(1, int(round(xp.get('min_child_weight', 5) * 0.67)))
    o[site] = {'xgb_params': xp}
json.dump(o, open('$TMPJSON','w'))
"
    export DATECT_HPARAM_OVERRIDE_JSON=$TMPJSON
    OUTDIR=candidate_verify_results/A_mcw067
else
    # Candidate B: smaller spike classifier (d3, n_est=200, no class weighting)
    TMPJSON=/tmp/cand_b_${SLURM_ARRAY_JOB_ID}_${TASK}.json
    $PYBIN -c "
import json
json.dump({
  'max_depth': 3, 'n_estimators': 200, 'scale_pos_weight': 1,
  'learning_rate': 0.05, 'subsample': 0.85, 'colsample_bytree': 0.85,
  'reg_alpha': 0.5, 'reg_lambda': 1.0, 'gamma': 0.1, 'min_child_weight': 3
}, open('$TMPJSON', 'w'))
"
    export DATECT_SPIKE_CLASSIFIER_JSON=$TMPJSON
    OUTDIR=candidate_verify_results/B_smaller_spike_clf
fi

mkdir -p $OUTDIR
echo "== Started: $(date) — task=$TASK seed=$SEED outdir=$OUTDIR =="
$PYBIN scripts/eval/multi_seed_baseline.py --seed $SEED --out-dir $OUTDIR
echo "== Done: $(date) =="

#!/bin/bash
#SBATCH --job-name=xgb-verify
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/xgb-verify-%A_%a.out
#SBATCH --error=logs/xgb-verify-%A_%a.err

set -euo pipefail
cd /gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid
mkdir -p logs xgb_verify_results

PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH
export DATECT_MIN_TRAINING_FOR_TUNING=99999

# 5 perturbations applied to ALL sites: 0=baseline, 1=depth+1, 2=depth-1, 3=mcw×1.5, 4=mcw×0.67
PERT=$SLURM_ARRAY_TASK_ID
TMPJSON=/tmp/xgb_verify_${SLURM_ARRAY_JOB_ID}_${PERT}.json

$PYBIN -c "
import json, sys; sys.path.insert(0, '.')
from forecasting.per_site_models import SITE_SPECIFIC_CONFIGS
o = {}
pert = $PERT
for site, cfg in SITE_SPECIFIC_CONFIGS.items():
    if cfg.get('xgb_params') is None: continue
    xp = dict(cfg['xgb_params'])
    if pert == 0:
        pass  # baseline
    elif pert == 1:
        xp['max_depth'] = max(2, xp.get('max_depth', 6) + 1)
    elif pert == 2:
        xp['max_depth'] = max(2, xp.get('max_depth', 6) - 1)
    elif pert == 3:
        xp['min_child_weight'] = max(1, int(round(xp.get('min_child_weight', 5) * 1.5)))
    elif pert == 4:
        xp['min_child_weight'] = max(1, int(round(xp.get('min_child_weight', 5) * 0.67)))
    o[site] = {'xgb_params': xp}
json.dump(o, open('$TMPJSON','w'))
"
export DATECT_HPARAM_OVERRIDE_JSON=$TMPJSON

LABELS=("baseline" "depth+1" "depth-1" "mcw_x1.5" "mcw_x0.67")
LABEL=${LABELS[$PERT]}

echo "== Started: $(date) — perturbation=$LABEL =="
$PYBIN scripts/eval/multi_seed_baseline.py --seed 42 --out-dir xgb_verify_results/p${PERT}_${LABEL}
echo "== Done: $(date) =="

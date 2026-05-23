#!/bin/bash
#SBATCH --job-name=pgrid-sweep
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/pgrid-sweep-%A_%a.out
#SBATCH --error=logs/pgrid-sweep-%A_%a.err

set -euo pipefail
cd /gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid
mkdir -p logs pgrid_sweep_results

PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH

# Inner-tune param grid alternatives (the global fallback). Configs:
# 0=current (depth4_n500 + depth6_n400), 1=shallow-only, 2=mid-only, 3=wider
# These override config.PARAM_GRID via an env var the engine reads at import.
# Note: we can't easily override PARAM_GRID via env, so just sweep at the per-site
# inner-tune level via DATECT_HPARAM_OVERRIDE_JSON's param_grid hook for all sites.
declare -a CFGS=("current" "shallow_only" "mid_only" "wider")

CFG=${CFGS[$SLURM_ARRAY_TASK_ID]}
TMPJSON=/tmp/pgrid_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json

$PYBIN -c "
import json, sys; sys.path.insert(0, '.')
from forecasting.per_site_models import SITE_SPECIFIC_CONFIGS
cfg = '$CFG'
new_grids = {
  'current': None,  # baseline — no override
  'shallow_only': [{'max_depth':3,'n_estimators':200,'learning_rate':0.05,'min_child_weight':5},
                    {'max_depth':2,'n_estimators':150,'learning_rate':0.03,'min_child_weight':10}],
  'mid_only': [{'max_depth':4,'n_estimators':300,'learning_rate':0.05,'min_child_weight':5},
                {'max_depth':5,'n_estimators':400,'learning_rate':0.05,'min_child_weight':5}],
  'wider': [{'max_depth':3,'n_estimators':200,'learning_rate':0.05,'min_child_weight':5},
             {'max_depth':5,'n_estimators':400,'learning_rate':0.05,'min_child_weight':3},
             {'max_depth':6,'n_estimators':500,'learning_rate':0.03,'min_child_weight':5}],
}
o = {}
if cfg != 'current':
    for site in SITE_SPECIFIC_CONFIGS:
        o[site] = {'param_grid': new_grids[cfg]}
json.dump(o, open('$TMPJSON','w'))
"
export DATECT_HPARAM_OVERRIDE_JSON=$TMPJSON

echo "== Started: $(date) — param_grid=$CFG =="
$PYBIN scripts/eval/multi_seed_baseline.py --seed 42 --out-dir pgrid_sweep_results/${CFG}
echo "== Done: $(date) =="

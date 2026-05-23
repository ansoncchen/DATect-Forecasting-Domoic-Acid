#!/bin/bash
#SBATCH --job-name=spike-clf-sweep
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/spike-clf-sweep-%A_%a.out
#SBATCH --error=logs/spike-clf-sweep-%A_%a.err

set -euo pipefail
cd /gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid
mkdir -p logs spike_clf_sweep_results

PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH
export DATECT_MIN_TRAINING_FOR_TUNING=99999

# 6 configs over (max_depth, n_estimators, scale_pos_weight). scale_pos_weight=3-5
# compensates for ~10% positive class (key F2 lever).
declare -a DEPTHS=(3 4 5 4 4 5)
declare -a NESTS=(200 300 400 300 300 400)
declare -a SPWS=(1 1 1 3 5 3)

D=${DEPTHS[$SLURM_ARRAY_TASK_ID]}
N=${NESTS[$SLURM_ARRAY_TASK_ID]}
SPW=${SPWS[$SLURM_ARRAY_TASK_ID]}

TMPJSON=/tmp/spike_clf_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json
$PYBIN -c "
import json
json.dump({
  'max_depth': $D, 'n_estimators': $N, 'scale_pos_weight': $SPW,
  'learning_rate': 0.05, 'subsample': 0.85, 'colsample_bytree': 0.85,
  'reg_alpha': 0.5, 'reg_lambda': 1.0, 'gamma': 0.1, 'min_child_weight': 3
}, open('$TMPJSON', 'w'))
"
export DATECT_SPIKE_CLASSIFIER_JSON=$TMPJSON

echo "== Started: $(date) — depth=$D n_est=$N spw=$SPW =="
$PYBIN scripts/eval/multi_seed_baseline.py --seed 42 --out-dir spike_clf_sweep_results/d${D}_n${N}_spw${SPW}
echo "== Done: $(date) =="

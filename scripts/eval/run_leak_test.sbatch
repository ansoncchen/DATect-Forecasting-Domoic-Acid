#!/bin/bash
#SBATCH --job-name=leak-test
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/leak-test-%A_%a.out
#SBATCH --error=logs/leak-test-%A_%a.err

set -euo pipefail
cd /gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid
mkdir -p logs leak_test_results
PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH
export DATECT_MIN_TRAINING_FOR_TUNING=99999
SEED=$((42 + SLURM_ARRAY_TASK_ID))
export DATECT_HPARAM_OVERRIDE_JSON=/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid/grid_winners_overrides.json
echo "== seed=$SEED grid-winners overrides =="
$PYBIN scripts/eval/multi_seed_baseline.py --seed $SEED --out-dir leak_test_results

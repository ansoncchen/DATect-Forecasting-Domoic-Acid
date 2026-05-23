#!/bin/bash
#SBATCH --job-name=mintrain-sweep
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/mintrain-sweep-%A_%a.out
#SBATCH --error=logs/mintrain-sweep-%A_%a.err

set -euo pipefail
cd /gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid
mkdir -p logs mintrain_sweep_results

PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH

VALUES=(40 60 80 120 160)
MIN_TRAIN=${VALUES[$SLURM_ARRAY_TASK_ID]}
export DATECT_MIN_TRAINING_FOR_TUNING=$MIN_TRAIN

echo "== Started: $(date) — MIN_TRAINING_FOR_TUNING=$MIN_TRAIN =="
$PYBIN scripts/eval/multi_seed_baseline.py --seed 42 --out-dir mintrain_sweep_results/min${MIN_TRAIN}
echo "== Done: $(date) =="

#!/bin/bash
#SBATCH --job-name=paper-metrics
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/paper-metrics-%j.out
#SBATCH --error=logs/paper-metrics-%j.err

set -euo pipefail
cd /gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid
mkdir -p logs eval_results
PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH
export DATECT_MIN_TRAINING_FOR_TUNING=99999
echo "== Started: $(date) =="
$PYBIN scripts/eval/eval_paper_metrics.py --sample-fraction 0.40 --seed 123 --force-rerun
echo "== Done: $(date) =="

#!/bin/bash
#SBATCH --job-name=oad-spike-tune
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/oad-spike-tune-%j.out
#SBATCH --error=logs/oad-spike-tune-%j.err

set -euo pipefail

REPO_ROOT="/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid"
cd "$REPO_ROOT" || { echo "REPO_ROOT not found: $REPO_ROOT"; exit 1; }
mkdir -p logs spike_tuning_results

PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH

N_TRIALS="${N_TRIALS:-40}"

echo "== Node: $(hostname) =="
echo "== N_TRIALS: $N_TRIALS =="
echo "== Started: $(date) =="

$PYBIN scripts/eval/tune_spike_classifier.py --n-trials "$N_TRIALS"

echo "== Done: $(date) =="

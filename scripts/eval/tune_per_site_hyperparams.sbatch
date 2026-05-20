#!/bin/bash
#SBATCH --job-name=oad-tune
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/oad-tune-%A_%a.out
#SBATCH --error=logs/oad-tune-%A_%a.err

set -euo pipefail

REPO_ROOT="/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid"
cd "$REPO_ROOT" || { echo "REPO_ROOT not found: $REPO_ROOT"; exit 1; }
mkdir -p logs tuning_results

PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH

N_TRIALS="${N_TRIALS:-30}"

echo "== Node: $(hostname) =="
echo "== Array task: $SLURM_ARRAY_TASK_ID =="
echo "== N_TRIALS: $N_TRIALS =="
echo "== Started: $(date) =="

$PYBIN scripts/eval/tune_per_site_hyperparams.py --n-trials "$N_TRIALS"

echo "== Done: $(date) =="

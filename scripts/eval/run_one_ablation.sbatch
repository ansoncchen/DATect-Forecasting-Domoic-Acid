#!/bin/bash
#SBATCH --job-name=oad-abl-arr
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=logs/oad-abl-arr-%A_%a.out
#SBATCH --error=logs/oad-abl-arr-%A_%a.err

set -euo pipefail

REPO_ROOT="/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid"
cd "$REPO_ROOT" || { echo "REPO_ROOT not found: $REPO_ROOT"; exit 1; }
mkdir -p logs partial_ablation

PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH

SEED="${SEED:-123}"

echo "== Node: $(hostname) =="
echo "== Array task: $SLURM_ARRAY_TASK_ID =="
echo "== Seed: $SEED =="
echo "== Started: $(date) =="

$PYBIN scripts/eval/run_one_ablation.py --seed "$SEED"

echo "== Done: $(date) =="

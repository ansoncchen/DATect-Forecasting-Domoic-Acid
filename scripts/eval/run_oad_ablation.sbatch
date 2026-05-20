#!/bin/bash
#SBATCH --job-name=oad-ablation
#SBATCH --partition=compute
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/oad-ablation-%j.out
#SBATCH --error=logs/oad-ablation-%j.err

set -euo pipefail

REPO_ROOT="/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid"
cd "$REPO_ROOT" || { echo "REPO_ROOT not found: $REPO_ROOT"; exit 1; }
mkdir -p logs

PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH

SEED="${SEED:-123}"

echo "== Node: $(hostname) =="
echo "== Seed: $SEED =="
echo "== Parquet: $(ls -lh data/processed/final_output.parquet | head -1) =="
echo "== Started: $(date) =="

$PYBIN scripts/eval/paper_ablation_study.py --seed "$SEED"

echo "== Done: $(date) =="

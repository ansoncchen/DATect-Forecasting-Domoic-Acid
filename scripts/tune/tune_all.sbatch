#!/bin/bash
#SBATCH --job-name=tune-all
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/tune-all-%j.out
#SBATCH --error=logs/tune-all-%j.err

# This is the orchestrator wrapper — it submits the 5 sweep jobs and waits for
# them, then assembles the JSON. Resource request is tiny because tune_all.py
# itself doesn't compute anything heavy; it only invokes sbatch and polls
# squeue. The actual compute happens in the 5 sweep jobs it submits.

set -euo pipefail
cd /gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid
mkdir -p logs config eval_outputs

PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH

echo "== Started: $(date) =="
$PYBIN scripts/tune/tune_all.py
echo "== Done: $(date) =="

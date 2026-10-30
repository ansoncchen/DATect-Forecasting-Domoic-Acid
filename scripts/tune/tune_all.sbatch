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
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"  # run the checkout this was submitted from
mkdir -p logs config results/tuning_provenance

PYBIN=${DATECT_PY:-/gscratch/stf/ac283/envs/datect_scratch/bin/python}
export PATH="$(dirname "$PYBIN"):$PATH"

echo "== Started: $(date) =="
$PYBIN scripts/tune/tune_all.py
echo "== Done: $(date) =="

#!/bin/bash
#SBATCH --job-name=datect-ablation
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=ablation-%j.out
#SBATCH --error=ablation-%j.err
#
# Paper ablation study (seed 42, 20% sample): baseline vs. no interpolated
# training, no per-site customization, no observation-order lags, no derived
# features, no OAD features. Writes results/ablation/paper_ablation_results.json.
# Submit in a disposable worktree:
#   bash scripts/eval/golden_baseline.sh <ref> <label> scripts/eval/paper_ablation_study.sbatch
set -euo pipefail

PY=${DATECT_PY:-/gscratch/stf/ac283/envs/datect_scratch/bin/python}
export TMPDIR=/gscratch/scrubbed/$USER/tmp
mkdir -p "$TMPDIR" results/ablation

git rev-parse HEAD > ablation_sha.txt
echo "== ablation @ $(date +%T) =="
$PY scripts/eval/paper_ablation_study.py --seed 42
echo "== done @ $(date +%T) =="

#!/bin/bash
#SBATCH --job-name=datect-cache
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=cache-%j.out
#SBATCH --error=cache-%j.err
#
# Pre-compute the dashboard cache (./cache) on Hyak; too heavy for a laptop.
# Submit in a disposable worktree, then copy cache/ next to the deploy:
#   bash scripts/eval/golden_baseline.sh <ref> cache-<label> deploy/precompute_cache.sbatch
#   scp -r klone-login:/gscratch/scrubbed/$USER/golden/cache-<label>/cache ./cache
set -euo pipefail

PY=${DATECT_PY:-/gscratch/stf/ac283/envs/datect_scratch/bin/python}
export TMPDIR=/gscratch/scrubbed/$USER/tmp
mkdir -p "$TMPDIR"

git rev-parse HEAD > cache_sha.txt
echo "== precompute @ $(date +%T) =="
$PY precompute_cache.py
echo "== done @ $(date +%T) =="

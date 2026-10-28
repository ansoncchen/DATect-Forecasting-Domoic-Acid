#!/bin/bash
#SBATCH --job-name=datect-golden
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=golden-%j.out
#SBATCH --error=golden-%j.err
#
# Runs every evaluation that backs a reported number, with the headline
# config (per-anchor tuning disabled). Submit via golden_baseline.sh so the
# job runs in a disposable worktree; outputs land at their usual relative paths.
set -euo pipefail

PY=${DATECT_PY:-/gscratch/stf/ac283/envs/datect_scratch/bin/python}
export DATECT_MIN_TRAINING_FOR_TUNING=99999
export TMPDIR=/gscratch/scrubbed/$USER/tmp
mkdir -p "$TMPDIR"

step() { echo "== $1 @ $(date +%T) =="; }

git rev-parse HEAD > golden_sha.txt
nproc > golden_nproc.txt

step "chronological validation window"
$PY scripts/eval/chronological_eval.py --window 2019-01-01:2022-01-01

step "chronological holdout window"
$PY scripts/eval/chronological_eval.py --window 2022-01-01:2024-01-01

for seed in 42 43 44 45 46; do
    step "multi-seed $seed"
    $PY scripts/eval/multi_seed_baseline.py --seed "$seed" --out-dir golden_multi_seed
done

step "paper metrics (seed 123, 40%)"
$PY scripts/eval/eval_paper_metrics.py --seed 123 --sample-fraction 0.40 --force-rerun

step "spike detection (reuses the seed-123 retrospective cache)"
$PY scripts/eval/spike_detection_eval.py --seed 123 --sample-fraction 0.40

step "done"

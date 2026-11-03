#!/bin/bash
# Onset round 2 (docs/FORECAST_EXPERIMENTS.md §6) on Hyak, run from node-local disk.
#
# /gscratch is nearly full and refuses writes now and then, so the job copies a
# `git archive` of this branch (unpacked in $ONSET_SRC, default ~/onset-src) to the
# node's /tmp, writes there, and copies the prediction parquets to
# $ONSET_OUT/<job id> (default ~/onset-results) on exit, including on preemption.
#
#   git archive HEAD | ssh klone-login 'rm -rf ~/onset-src && mkdir -p ~/onset-src && tar -x -C ~/onset-src'
#   ssh klone-login 'cd ~/onset-src && VARIANTS=all sbatch scripts/experiments/run_onset.sbatch'
#
#SBATCH --job-name=fx-onset
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fx_onset_%j.out
set -euo pipefail
PY=${DATECT_PY:-/gscratch/stf/ac283/envs/datect_scratch/bin/python}
SRC=${ONSET_SRC:-$HOME/onset-src}
JOB=/tmp/onset-$SLURM_JOB_ID
OUT=${ONSET_OUT:-$HOME/onset-results}/$SLURM_JOB_ID

mkdir -p "$JOB" "$OUT"
cp -r "$SRC/." "$JOB/"
finish() {
    cp -r "$JOB/results/experiments/onset_raw/." "$OUT/" 2>/dev/null || true
    rm -rf "$JOB"
    echo "== results in $OUT @ $(date +%T) =="
}
trap finish EXIT
trap 'exit 143' TERM

cd "$JOB"
export MPLBACKEND=Agg TZ=UTC
echo "== $(cat REVISION 2>/dev/null || echo unknown) on $(hostname), $(nproc) cpus @ $(date +%T) =="
"$PY" scripts/experiments/onset_study.py --variants "${VARIANTS:-all}" \
    --start "${START:-2015-01-01}" --end "${END:-2022-01-01}" \
    --out-dir results/experiments/onset_raw --verify
echo "done $(date)"

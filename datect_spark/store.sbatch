#!/bin/bash
#SBATCH --job-name=datect-store
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/store-%j.out
#SBATCH --error=logs/store-%j.err
#
# The precomputed store's gate: the API's precomputed endpoints must return the
# same bodies from the Delta store as from the file cache it replaces.
#   before  OLD_SHA's precompute_cache.py (joblib) -> cache/, then its API's bodies
# Runs on node-local disk (hyak_stage.sh); results/store/*.json is kept.
#   sbatch -o ~/datect-spark-logs/%x-%j.out -e ~/datect-spark-logs/%x-%j.err \
#       --export=ALL,STEPS=before,OLD_SHA=<sha> datect_spark/store.sbatch
set -euo pipefail

export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:////gscratch/stf/ac283/datect-mlflow/mlflow.db}
RESULTS_SUBDIR=store
source "$SLURM_SUBMIT_DIR/datect_spark/hyak_stage.sh"
mkdir -p results/store

for step in ${STEPS:-before}; do
    echo "== $step @ $(date +%T) =="
    case $step in
        before) archive "${OLD_SHA:?set OLD_SHA}" "$JOB_DIR/old"
                (cd "$JOB_DIR/old" && /usr/bin/time -v $PY -u precompute_cache.py) 2>&1 | tail -40
                ENABLE_PRECOMPUTED_CACHE=true CACHE_DIR=$JOB_DIR/old/cache \
                    $PY -u scripts/eval/endpoint_bodies.py --repo "$JOB_DIR/old" \
                    --out results/store/bodies_before.json ;;
        *) echo "unknown step $step"; exit 1 ;;
    esac
done
echo "== done @ $(date +%T) =="

#!/bin/bash
#SBATCH --job-name=datect-serving
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --output=logs/serving-%j.out
#SBATCH --error=logs/serving-%j.err
#
# Precomputed forecasts served from Delta, on one node: latency of the API at
# MAIN_SHA (before), the Spark batch inference with its MLflow registry entry,
# the parity check, and latency of this commit's API reading the table (after).
# STEPS picks what runs, in order. The job runs on node-local disk (see
# hyak_stage.sh); results/serving/*.json is kept, and the table is copied to
# scrubbed when that succeeds. The API under test runs with the job's threads (no
# OMP_NUM_THREADS pin), as on a 16-core server; Spark workers pin one thread each.
#   sbatch -o ~/datect-spark-logs/%x-%j.out -e ~/datect-spark-logs/%x-%j.err \
#       --export=ALL,MAIN_SHA=<main sha> datect_spark/serving.sbatch
#   sbatch ... --export=ALL,STEPS="precompute parity",LIMIT=24,MLFLOW_TRACKING_URI=local datect_spark/serving.sbatch  # smoke
set -euo pipefail

export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:////gscratch/stf/ac283/datect-mlflow/mlflow.db}
RESULTS_SUBDIR=serving
source "$SLURM_SUBMIT_DIR/datect_spark/hyak_stage.sh"
mkdir -p results/serving

for step in ${STEPS:-latency-before precompute parity latency-after}; do
    echo "== $step @ $(date +%T) =="
    case $step in
        latency-before) archive "${MAIN_SHA:?set MAIN_SHA}" "$JOB_DIR/main"
                        $PY -u scripts/eval/forecast_latency.py --repo "$JOB_DIR/main" \
                            --out results/serving/latency_before.json ;;
        precompute)     $PY -u -m datect_spark.forecast_serving precompute ${GRID:+--grid $GRID} ${LIMIT:+--limit $LIMIT} ;;
        parity)         $PY -u -m datect_spark.forecast_serving parity ;;
        latency-after)  DATECT_FORECAST_TABLE=$DATECT_LAKE/gold/forecasts \
                            $PY -u scripts/eval/forecast_latency.py --repo . \
                            --out results/serving/latency_after.json \
                            ${MAIN_SHA:+--compare-to results/serving/latency_before.json} ;;
        *) echo "unknown step $step"; exit 1 ;;
    esac
done
if [ -d "$DATECT_LAKE/gold/forecasts" ]; then
    dest=/gscratch/scrubbed/ac283/datect-lake/lake/gold/forecasts-$SLURM_JOB_ID
    cp -r "$DATECT_LAKE/gold/forecasts" "$dest" && echo "table copied to $dest" || echo "table copy failed"
fi
echo "== done @ $(date +%T) =="

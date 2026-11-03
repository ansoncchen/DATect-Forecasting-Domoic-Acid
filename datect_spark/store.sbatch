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
# Build the API's precomputed store (Delta tables in $DATECT_LAKE/gold) on Spark and
# check it. STEPS picks what runs, in order:
#   forecasts      forecast_serving precompute (gold/forecasts, MLflow registry entry)
#   parity         50 forecasts vs a fresh on-demand computation
#   retrospective  precompute_store retrospective (8 slices; joblib -> Spark)
#   visuals        precompute_store visuals (spectral + correlation)
#   after          the API's precomputed-endpoint bodies from the store, compared with
#                  BEFORE_JSON (a `before` run)
#   before         OLD_SHA's precompute_cache.py (joblib) -> cache/, then its API's bodies
#   save           copy the store's tables to $DATECT_STORE_OUT/<sha>/ (the deploy input)
# Runs on node-local disk (hyak_stage.sh); results/store/*.json is kept. The forecasts
# and the other tables can be built by two jobs on two nodes, each saving its tables.
#   sbatch -o ~/datect-spark-logs/%x-%j.out -e ~/datect-spark-logs/%x-%j.err \
#       --export=ALL,STEPS="forecasts parity save" datect_spark/store.sbatch
#   sbatch ... --export=ALL,STEPS="retrospective visuals after save",BEFORE_JSON=<path> datect_spark/store.sbatch
set -euo pipefail

export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:////gscratch/stf/ac283/datect-mlflow/mlflow.db}
RESULTS_SUBDIR=store
source "$SLURM_SUBMIT_DIR/datect_spark/hyak_stage.sh"
mkdir -p results/store

for step in ${STEPS:-forecasts parity retrospective visuals save}; do
    echo "== $step @ $(date +%T) =="
    case $step in
        forecasts)     $PY -u -m datect_spark.forecast_serving precompute ${GRID:+--grid $GRID} ${LIMIT:+--limit $LIMIT}
                       mv results/serving/precompute.json results/store/precompute_forecasts.json ;;
        parity)        $PY -u -m datect_spark.forecast_serving parity
                       mv results/serving/parity.json results/store/parity_forecasts.json ;;
        retrospective) $PY -u -m datect_spark.precompute_store retrospective ;;
        visuals)       $PY -u -m datect_spark.precompute_store visuals ;;
        after)         DATECT_PRECOMPUTED_DIR=$DATECT_LAKE/gold \
                           $PY -u scripts/eval/endpoint_bodies.py --repo . \
                           --out results/store/bodies_after.json \
                           ${BEFORE_JSON:+--compare-to $BEFORE_JSON} ;;
        before)        archive "${OLD_SHA:?set OLD_SHA}" "$JOB_DIR/old"
                       (cd "$JOB_DIR/old" && /usr/bin/time -v $PY -u precompute_cache.py) 2>&1 | tail -40
                       ENABLE_PRECOMPUTED_CACHE=true CACHE_DIR=$JOB_DIR/old/cache \
                           $PY -u scripts/eval/endpoint_bodies.py --repo "$JOB_DIR/old" \
                           --out results/store/bodies_before.json ;;
        save)          dest=${DATECT_STORE_OUT:-$HOME/datect-store}/$SHA
                       mkdir -p "$dest"
                       for t in forecasts retrospective spectral correlation; do
                           [ -d "$DATECT_LAKE/gold/$t" ] && cp -r "$DATECT_LAKE/gold/$t" "$dest/" && echo "saved $t -> $dest"
                       done; du -sh "$dest" ;;
        *) echo "unknown step $step"; exit 1 ;;
    esac
done
echo "== done @ $(date +%T) =="

#!/bin/bash
#SBATCH --job-name=datect-serving
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/serving-%j.out
#SBATCH --error=logs/serving-%j.err
#
# Precomputed forecasts served from Delta: latency before/after, the Spark batch
# inference, the MLflow registry entry and the parity check. STEPS picks what runs,
# in order. Submit from the repo root of a disposable worktree:
#   sbatch --export=ALL,STEPS="latency-before",MAIN_REPO=/path/to/main-worktree datect_spark/serving.sbatch
#   sbatch --export=ALL,STEPS="precompute parity latency-after" datect_spark/serving.sbatch
#
# The API under test runs with the job's threads (no OMP_NUM_THREADS pin), as it
# would on a 16-core server; Spark workers pin one thread per task themselves.
set -euo pipefail

module load coenv/jdk/17.0.8
PY=${DATECT_SPARK_PY:-/gscratch/scrubbed/ac283/envs/datect_spark/bin/python}
ROOT=${DATECT_LAKE_ROOT:-/gscratch/scrubbed/ac283/datect-lake}
export DATECT_LAKE=$ROOT/lake
export DATECT_IVY_DIR=$ROOT/ivy
export SPARK_LOCAL_DIRS=$ROOT/tmp
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:////gscratch/stf/ac283/datect-mlflow/mlflow.db}
export SPARK_MASTER="local[${SLURM_CPUS_PER_TASK:-8}]"
export TZ=UTC
mkdir -p "$DATECT_LAKE" "$SPARK_LOCAL_DIRS" /gscratch/stf/ac283/datect-mlflow results/serving

git rev-parse HEAD > results/serving/serving_sha.txt
for step in ${STEPS:-precompute parity latency-after}; do
    echo "== $step @ $(date +%T) =="
    case $step in
        latency-before) $PY -u scripts/eval/forecast_latency.py --repo "${MAIN_REPO:?set MAIN_REPO}" \
                            --out results/serving/latency_before.json ;;
        precompute)     $PY -u -m datect_spark.forecast_serving precompute ${GRID:+--grid $GRID} ${LIMIT:+--limit $LIMIT} ;;
        parity)         $PY -u -m datect_spark.forecast_serving parity ;;
        latency-after)  DATECT_FORECAST_TABLE=$DATECT_LAKE/gold/forecasts \
                            $PY -u scripts/eval/forecast_latency.py --repo . \
                            --out results/serving/latency_after.json \
                            --compare-to results/serving/latency_before.json ;;
        *) echo "unknown step $step"; exit 1 ;;
    esac
done
echo "== done @ $(date +%T) =="

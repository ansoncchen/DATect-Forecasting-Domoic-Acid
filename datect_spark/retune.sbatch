#!/bin/bash
#SBATCH --job-name=datect-retune
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=retune-%j.out
#SBATCH --error=retune-%j.err
#
# Weights x clip re-tune on Spark (local mode on one node) with Delta + MLflow.
# STEPS picks what runs, in order (default: all four). Every step overwrites
# its outputs, so a ckpt requeue just repeats it. Submit from the repo root of
# a disposable worktree:
#   sbatch datect_spark/retune.sbatch
#   sbatch --export=ALL,STEPS="predict-tuning" datect_spark/retune.sbatch
#
# Storage: the lake, Spark scratch and MLflow artifacts live on scrubbed (the
# stf allocation is nearly full); only the small MLflow database is on stf.
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
mkdir -p "$DATECT_LAKE" "$SPARK_LOCAL_DIRS" /gscratch/stf/ac283/datect-mlflow

git rev-parse HEAD > retune_sha.txt
for step in ${STEPS:-predict-tuning predict-holdout score holdout}; do
    echo "== $step @ $(date +%T) =="
    case $step in
        predict-tuning)     $PY -u -m datect_spark.retune predict --window tuning ${LIMIT:+--limit $LIMIT} ;;
        predict-holdout)    $PY -u -m datect_spark.retune predict --window holdout ${LIMIT:+--limit $LIMIT} ;;
        score)              $PY -u -m datect_spark.retune score ;;
        holdout)            $PY -u -m datect_spark.retune holdout ;;
        *) echo "unknown step $step"; exit 1 ;;
    esac
done
echo "== done @ $(date +%T) =="

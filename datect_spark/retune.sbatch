#!/bin/bash
#SBATCH --job-name=datect-retune
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=logs/retune-%j.out
#SBATCH --error=logs/retune-%j.err
#
# Per-site ensemble-weight re-tune on Spark (local mode on one node) with Delta +
# MLflow. STEPS picks what runs, in order (default: all four). The job runs on
# node-local disk (see hyak_stage.sh); results/tuning_provenance/retune_weights is
# kept. Nothing is adopted: proposed_overrides.json is a proposal.
#   sbatch -o ~/datect-spark-logs/%x-%j.out -e ~/datect-spark-logs/%x-%j.err datect_spark/retune.sbatch
#   sbatch ... --export=ALL,STEPS="predict-holdout",LIMIT=20,MLFLOW_TRACKING_URI=local datect_spark/retune.sbatch  # smoke
set -euo pipefail

export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:////gscratch/stf/ac283/datect-mlflow/mlflow.db}
RESULTS_SUBDIR=tuning_provenance/retune_weights
source "$SLURM_SUBMIT_DIR/datect_spark/hyak_stage.sh"
mkdir -p "results/$RESULTS_SUBDIR"

for step in ${STEPS:-predict-tuning predict-holdout score holdout}; do
    echo "== $step @ $(date +%T) =="
    case $step in
        predict-tuning)  $PY -u -m datect_spark.retune predict --window tuning ${LIMIT:+--limit $LIMIT} ;;
        predict-holdout) $PY -u -m datect_spark.retune predict --window holdout ${LIMIT:+--limit $LIMIT} ;;
        score)           $PY -u -m datect_spark.retune score ;;
        holdout)         $PY -u -m datect_spark.retune holdout ;;
        *) echo "unknown step $step"; exit 1 ;;
    esac
done
echo "== done @ $(date +%T) =="

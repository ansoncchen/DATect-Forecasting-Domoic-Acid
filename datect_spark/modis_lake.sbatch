#!/bin/bash
#SBATCH --job-name=datect-modis-lake
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/modis-lake-%j.out
#SBATCH --error=logs/modis-lake-%j.err
#
# MODIS NetCDFs -> bronze/silver/gold Delta tables (Spark local mode on one node),
# then exports data/processed/oad_cloud_fractions.parquet (results/spark_lake/), or the
# timed reference (02_build_cube.py + reference_valid_fractions).
# STEPS picks what runs, in order. The job runs on node-local disk (see
# hyak_stage.sh); the lake is rebuilt each run and only results/spark_lake/*.json
# is kept. Submit from any checkout, with logs outside the full filesystems:
#   sbatch -o ~/datect-spark-logs/%x-%j.out -e ~/datect-spark-logs/%x-%j.err datect_spark/modis_lake.sbatch
#   sbatch ... --export=ALL,STEPS=reference,MLFLOW_TRACKING_URI=sqlite:////gscratch/stf/ac283/datect-mlflow/modis_reference.db datect_spark/modis_lake.sbatch
#   sbatch ... --export=ALL,DATECT_MODIS_YEARS=2015,MLFLOW_TRACKING_URI=local datect_spark/modis_lake.sbatch   # smoke
set -euo pipefail

export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:////gscratch/stf/ac283/datect-mlflow/modis_lake.db}
RESULTS_SUBDIR=spark_lake
source "$SLURM_SUBMIT_DIR/datect_spark/hyak_stage.sh"

for step in ${STEPS:-bronze silver optimize gold parity export vacuum}; do
    echo "== $step @ $(date +%T) =="
    $PY -u -m datect_spark.modis_lake "$step"
    echo "   lake on disk: $(du -sh "$DATECT_LAKE" | cut -f1); node /tmp free: $(df -h /tmp | tail -1 | awk '{print $4}')"
done
echo "== done @ $(date +%T) =="

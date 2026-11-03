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
# or the timed reference (02_build_cube.py + compute_region_valid_fractions).
# STEPS picks what runs, in order; a step already done at this commit is skipped,
# so a ckpt requeue resumes. Submit from the repo root of a disposable worktree:
#   sbatch datect_spark/modis_lake.sbatch
#   sbatch --export=ALL,STEPS=reference datect_spark/modis_lake.sbatch
#
# Storage: the lake, the reference cube and Spark scratch live on scrubbed (the
# stf allocation is nearly full); only the small MLflow database is on stf.
set -euo pipefail

module load coenv/jdk/17.0.8
PY=${DATECT_SPARK_PY:-/gscratch/scrubbed/ac283/envs/datect_spark/bin/python}
ROOT=${DATECT_LAKE_ROOT:-/gscratch/scrubbed/ac283/datect-lake}
export DATECT_LAKE=$ROOT/lake
export DATECT_IVY_DIR=${DATECT_IVY_DIR:-$ROOT/ivy}
export SPARK_LOCAL_DIRS=$ROOT/tmp
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:////gscratch/stf/ac283/datect-mlflow/mlflow.db}
export SPARK_MASTER="local[${SLURM_CPUS_PER_TASK:-8}]"
export MPLBACKEND=Agg
export TZ=UTC
mkdir -p "$DATECT_LAKE" "$SPARK_LOCAL_DIRS" /gscratch/stf/ac283/datect-mlflow

echo "== start @ $(date +%T) on $(hostname), $(nproc) cpus; scrubbed use: $(du -sh "$ROOT" | cut -f1) =="
for step in ${STEPS:-bronze silver optimize gold parity vacuum}; do
    echo "== $step @ $(date +%T) =="
    $PY -u -m datect_spark.modis_lake "$step"
    echo "   lake on disk: $(du -sh "$DATECT_LAKE" | cut -f1)"
done
echo "== done @ $(date +%T) =="

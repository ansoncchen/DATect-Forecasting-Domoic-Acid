# Sourced by datect_spark/*.sbatch: run a job from the node's local disk.
#
# /gscratch/scrubbed and /gscratch/stf are nearly full and refuse writes
# intermittently (EDQUOT: "Disk quota exceeded"), so a job reads its inputs there
# but writes only to the node's local disk (/tmp: 358 GB xfs, no quota). It runs
# from a `git archive` of the submitting checkout's HEAD (or $SHA), keeps the lake
# and Spark scratch beside it, and on exit copies results/$RESULTS_SUBDIR to
# $DATECT_RESULTS_OUT/<job id> (default ~/datect-spark-results) and deletes the
# directory. MLflow stays on stf (small sqlite writes); MLFLOW_TRACKING_URI=local
# keeps it on the node too (smoke runs).

module load coenv/jdk/17.0.8
PY=${DATECT_SPARK_PY:-/gscratch/scrubbed/ac283/envs/datect_spark/bin/python}
SRC=$SLURM_SUBMIT_DIR
SHA=${SHA:-$(git -C "$SRC" rev-parse HEAD)}
JOB_DIR=${DATECT_JOB_DIR:-/tmp/datect-$SLURM_JOB_ID}
RESULTS_OUT=${DATECT_RESULTS_OUT:-$HOME/datect-spark-results}/$SLURM_JOB_ID
IVY_SRC=${DATECT_IVY_SRC:-/gscratch/scrubbed/ac283/datect-lake/ivy}  # resolved Delta jars

archive() {  # <ref> <dir>: a checkout without .git; REVISION records the commit
    mkdir -p "$2"
    git -C "$SRC" archive "$1" | tar -x -C "$2"
    git -C "$SRC" rev-parse "$1" > "$2/REVISION"
}

finish() {
    mkdir -p "$RESULTS_OUT"
    cp -r "$JOB_DIR/repo/results/$RESULTS_SUBDIR" "$RESULTS_OUT/" 2>/dev/null || true
    rm -rf "$JOB_DIR"
    echo "== results in $RESULTS_OUT @ $(date +%T) =="
}

archive "$SHA" "$JOB_DIR/repo"
mkdir -p "$JOB_DIR/lake" "$JOB_DIR/tmp"
cp -r "$IVY_SRC" "$JOB_DIR/ivy"
trap finish EXIT
trap 'exit 143' TERM  # ckpt preemption: still copy what exists, then requeue

export DATECT_LAKE=$JOB_DIR/lake
export DATECT_IVY_DIR=$JOB_DIR/ivy
export SPARK_LOCAL_DIRS=$JOB_DIR/tmp
export TMPDIR=$JOB_DIR/tmp
export SPARK_MASTER="local[${SLURM_CPUS_PER_TASK:-8}]"
export MPLBACKEND=Agg
export TZ=UTC
if [ "${MLFLOW_TRACKING_URI:-}" = local ]; then
    export MLFLOW_TRACKING_URI=sqlite:///$JOB_DIR/mlflow.db
    export DATECT_MLFLOW_ARTIFACTS=$JOB_DIR/mlartifacts
fi
export DATECT_MLFLOW_ARTIFACTS=${DATECT_MLFLOW_ARTIFACTS:-/gscratch/stf/ac283/datect-mlflow/mlartifacts}
mkdir -p /gscratch/stf/ac283/datect-mlflow
cd "$JOB_DIR/repo"
echo "== $(cat REVISION) on $(hostname), $(nproc) cpus, job dir $JOB_DIR @ $(date +%T) =="

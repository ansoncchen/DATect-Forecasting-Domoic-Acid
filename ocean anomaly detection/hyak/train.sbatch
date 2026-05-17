#!/bin/bash
#SBATCH --job-name=oad-train
#SBATCH --partition=gpu-2080ti       # stf has 2 free GPUs here; for big runs use --export=ALL,PARTITION=ckpt
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/oad-train-%j.out
#SBATCH --error=logs/oad-train-%j.err

set -euo pipefail


# ----------------------------------------------------------------------------
# DATect / Ocean Anomaly Detection — training driver
#
# Edit OAD_ROOT below to your install path on Hyak.
# Submits with:  sbatch hyak/train.sbatch
# Override mode: sbatch --export=ALL,MODE=temporal_sweep hyak/train.sbatch
#                MODE ∈ {snapshot, snapshot_sweep, temporal, temporal_sweep,
#                        snapshot_ablate, temporal_ablate, smoke}
# ----------------------------------------------------------------------------

OAD_ROOT="/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid/ocean anomaly detection"
MODE="${MODE:-snapshot}"

cd "$OAD_ROOT" || { echo "OAD_ROOT not found: $OAD_ROOT"; exit 1; }
mkdir -p logs

# CUDA provided by conda env
# Use env's python directly (most robust)
PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH

echo "== MODE=$MODE =="
echo "== Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) =="
echo "== Cube: $(ls -lh data/cube.zarr 2>/dev/null | head -1) =="

case "$MODE" in
  smoke)
    $PYBIN scripts/03_train_ae.py --latent 32 --epochs 5 --debug
    $PYBIN scripts/03_train_ae.py --temporal --latent 32 --epochs 5 --debug
    ;;
  snapshot)
    $PYBIN scripts/03_train_ae.py --latent 32
    ;;
  snapshot_sweep)
    $PYBIN scripts/03_train_ae.py --sweep
    ;;
  snapshot_ablate)
    $PYBIN scripts/03_train_ae.py --ablate-channels
    ;;
  temporal)
    $PYBIN scripts/03_train_ae.py --temporal --latent 32
    ;;
  temporal_sweep)
    $PYBIN scripts/03_train_ae.py --temporal --sweep
    ;;
  temporal_ablate)
    $PYBIN scripts/03_train_ae.py --temporal --ablate-channels
    ;;
  *)
    echo "Unknown MODE: $MODE"; exit 1 ;;
esac

echo "== Done =="

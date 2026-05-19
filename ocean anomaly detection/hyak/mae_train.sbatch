#!/bin/bash
#SBATCH --job-name=oad-mae-train
#SBATCH --partition=ckpt
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx6k:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/oad-mae-train-%j.out
#SBATCH --error=logs/oad-mae-train-%j.err

set -euo pipefail

# ----------------------------------------------------------------------------
# Phase C MAE-only training driver — trains BOTH 2D ConvAE and 3D ConvAE3D
# at a single MASK_RATIO and LATENT_DIM. NO download, NO cube rebuild,
# NO inference, NO eval. Designed for chaining multiple MASK_RATIO jobs
# back-to-back without each one re-doing everything.
#
# Usage:
#   sbatch --export=ALL,MASK_RATIO=0.15 hyak/mae_train.sbatch
#   sbatch --export=ALL,MASK_RATIO=0.40,EPOCHS=100,LATENT=32 hyak/mae_train.sbatch
# ----------------------------------------------------------------------------

OAD_ROOT="/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid/ocean anomaly detection"
MASK_RATIO="${MASK_RATIO:?must set MASK_RATIO (e.g. 0.30)}"
EPOCHS="${EPOCHS:-100}"
LATENT="${LATENT:-32}"
SEED="${SEED:-42}"

cd "$OAD_ROOT"
mkdir -p logs

PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH

echo "============================================================="
echo "==  Phase C MAE training only                              =="
echo "==  Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "==  MASK_RATIO=$MASK_RATIO  LATENT=$LATENT  EPOCHS=$EPOCHS  SEED=$SEED"
echo "============================================================="
echo

if [ ! -d data/cube.zarr ]; then
  echo "ERROR: data/cube.zarr not found; build it first via hyak/build_cube.sbatch"
  exit 1
fi

echo "=== Train 2D ConvAE with mask_ratio=$MASK_RATIO ==="
$PYBIN scripts/03_train_ae.py --latent $LATENT --epochs $EPOCHS --seed $SEED \
    --mask-ratio $MASK_RATIO
echo

echo "=== Train 3D ConvAE3D with mask_ratio=$MASK_RATIO ==="
$PYBIN scripts/03_train_ae.py --temporal --latent $LATENT --epochs $EPOCHS --seed $SEED \
    --mask-ratio $MASK_RATIO
echo

echo "=== Checkpoints written ==="
ls -la models/*mae*$( printf "%03d" $(echo "$MASK_RATIO * 100" | bc | cut -d. -f1) )*.pt 2>&1 | tail
echo
echo "== DONE: mask_ratio=$MASK_RATIO =="

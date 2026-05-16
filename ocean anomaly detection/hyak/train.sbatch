#!/bin/bash
#SBATCH --job-name=oad-train
#SBATCH --partition=gpu-a40             # or gpu-rtx6k, gpu-2080ti — adjust to your allocation
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/oad-train-%j.out
#SBATCH --error=logs/oad-train-%j.err

# ----------------------------------------------------------------------------
# DATect / Ocean Anomaly Detection — training driver
#
# Edit OAD_ROOT below to your install path on Hyak.
# Submits with:  sbatch hyak/train.sbatch
# Override mode: sbatch --export=ALL,MODE=temporal_sweep hyak/train.sbatch
#                MODE ∈ {snapshot, snapshot_sweep, temporal, temporal_sweep,
#                        snapshot_ablate, temporal_ablate, smoke}
# ----------------------------------------------------------------------------

OAD_ROOT="/gscratch/stf/ac283/oad"
MODE="${MODE:-snapshot}"

cd "$OAD_ROOT" || { echo "OAD_ROOT not found: $OAD_ROOT"; exit 1; }
mkdir -p logs

module load cuda/12.4 2>/dev/null || true   # adjust to your cluster
source .venv/bin/activate

echo "== MODE=$MODE =="
echo "== Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) =="
echo "== Cube: $(ls -lh data/cube.zarr 2>/dev/null | head -1) =="

case "$MODE" in
  smoke)
    python scripts/03_train_ae.py --latent 32 --epochs 5 --debug
    python scripts/03_train_ae.py --temporal --latent 32 --epochs 5 --debug
    ;;
  snapshot)
    python scripts/03_train_ae.py --latent 32
    ;;
  snapshot_sweep)
    python scripts/03_train_ae.py --sweep
    ;;
  snapshot_ablate)
    python scripts/03_train_ae.py --ablate-channels
    ;;
  temporal)
    python scripts/03_train_ae.py --temporal --latent 32
    ;;
  temporal_sweep)
    python scripts/03_train_ae.py --temporal --sweep
    ;;
  temporal_ablate)
    python scripts/03_train_ae.py --temporal --ablate-channels
    ;;
  *)
    echo "Unknown MODE: $MODE"; exit 1 ;;
esac

echo "== Done =="

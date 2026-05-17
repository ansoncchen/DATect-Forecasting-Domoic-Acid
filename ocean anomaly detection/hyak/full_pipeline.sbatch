#!/bin/bash
#SBATCH --job-name=oad-full
#SBATCH --partition=gpu-2080ti       # stf has 2 free GPUs here; for big runs use --export=ALL,PARTITION=ckpt
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/oad-full-%j.out
#SBATCH --error=logs/oad-full-%j.err

# ----------------------------------------------------------------------------
# End-to-end pipeline: download → cube → train (2D+3D sweeps) → infer → eval.
# Assumes already-downloaded data; for fresh downloads use hyak/download.sbatch first.
#
# Submit:  sbatch hyak/full_pipeline.sbatch
# Skip download:  sbatch --export=ALL,SKIP_DOWNLOAD=1 hyak/full_pipeline.sbatch
# Custom range:   sbatch --export=ALL,START=2003-01-01,END=2010-12-31 hyak/full_pipeline.sbatch
# ----------------------------------------------------------------------------

OAD_ROOT="/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid/ocean anomaly detection"
START="${START:-2003-01-01}"
END="${END:-2024-12-31}"
CHANNELS="${CHANNELS:-chla,k490,nflh,sst}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
EPOCHS="${EPOCHS:-100}"
LATENT_SWEEP="${LATENT_SWEEP:-1}"        # 1=sweep, 0=single run at LATENT_DIM=32

cd "$OAD_ROOT" || { echo "OAD_ROOT not found: $OAD_ROOT"; exit 1; }
mkdir -p logs

# CUDA provided by conda env
source /gscratch/stf/ac283/envs/datect_scratch/bin/activate 2>/dev/null || conda activate /gscratch/stf/ac283/envs/datect_scratch

echo "============================================================="
echo "==  DATect / Ocean Anomaly Detection — full pipeline       =="
echo "==  Node: $(hostname)   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "==  Channels: $CHANNELS   Range: $START → $END"
echo "==  Epochs: $EPOCHS   Latent sweep: $LATENT_SWEEP"
echo "============================================================="
echo

# ----------------------------------------------------------------------------
# Stage 1: download (skippable)
# ----------------------------------------------------------------------------
if [ "$SKIP_DOWNLOAD" != "1" ]; then
  echo "=== Stage 1/5: download ==="
  python scripts/01_download.py --channels "$CHANNELS" --start "$START" --end "$END" --workers 8
  echo
fi

# ----------------------------------------------------------------------------
# Stage 2: build cube
# ----------------------------------------------------------------------------
echo "=== Stage 2/5: build cube ==="
python scripts/02_build_cube.py
ls -lh data/cube.zarr 2>&1 | head -3
echo

# ----------------------------------------------------------------------------
# Stage 3: train (2D + 3D, optionally sweep)
# ----------------------------------------------------------------------------
echo "=== Stage 3/5: train 2D AE ==="
if [ "$LATENT_SWEEP" = "1" ]; then
  python scripts/03_train_ae.py --sweep --epochs $EPOCHS
else
  python scripts/03_train_ae.py --latent 32 --epochs $EPOCHS
fi
echo

echo "=== Stage 3/5: train 3D ConvAE3D ==="
if [ "$LATENT_SWEEP" = "1" ]; then
  python scripts/03_train_ae.py --temporal --sweep --epochs $EPOCHS
else
  python scripts/03_train_ae.py --temporal --latent 32 --epochs $EPOCHS
fi
echo

# ----------------------------------------------------------------------------
# Stage 4: inference (all baselines + all checkpoints)
# ----------------------------------------------------------------------------
echo "=== Stage 4/5: inference (B1, B2, B3, B3T, all AE) ==="
python scripts/04_run_inference.py --all-ae --baselines-only --baselines-3d
echo

# ----------------------------------------------------------------------------
# Stage 5: evaluate
# ----------------------------------------------------------------------------
echo "=== Stage 5/5: evaluation figures + numeric summary ==="
python scripts/05_evaluate.py
echo
python scripts/06_summarize_results.py | tee outputs/figures/RESULTS_summary.txt
echo

echo "============================================================="
echo "== DONE — results in $OAD_ROOT/outputs/                    =="
echo "==   models/        — AE checkpoints                       =="
echo "==   outputs/scores/  — per-method score parquets          =="
echo "==   outputs/figures/  — all E1/E2/E4/E5 + sanity PNGs     =="
echo "==                      + RESULTS_summary.txt              =="
echo "============================================================="

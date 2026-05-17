#!/bin/bash
#SBATCH --job-name=oad-download
#SBATCH --partition=compute              # CPU-only — no GPU needed for downloads
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00                  # generous — ~24 yr × 4 ch × 364 days at ~5 s/file / 8 workers ≈ 6 hr
#SBATCH --output=logs/oad-download-%j.out
#SBATCH --error=logs/oad-download-%j.err

set -euo pipefail


# ----------------------------------------------------------------------------
# DATect / Ocean Anomaly Detection — download driver
#
# Downloads all 4 MODIS channels at stride-2 (0.025°) keeping ALL daily
# rolling 8-day composites (~364/yr/channel) for denser training signal.
# Matches the original ocean-anomaly branch behavior.
#
# Submit defaults (full 2003-2024 range):  sbatch hyak/download.sbatch
# Custom range:  sbatch --export=ALL,START=2003-01-01,END=2010-12-31 hyak/download.sbatch
# Single year:   sbatch --export=ALL,START=2003-01-01,END=2003-12-31 hyak/download.sbatch
# Channels:      sbatch --export=ALL,CHANNELS=chla,sst hyak/download.sbatch
# ----------------------------------------------------------------------------

OAD_ROOT="/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid/ocean anomaly detection"
START="${START:-2003-01-01}"
END="${END:-2024-12-31}"
CHANNELS="${CHANNELS:-chla,k490,nflh,sst}"

cd "$OAD_ROOT" || { echo "OAD_ROOT not found: $OAD_ROOT"; exit 1; }
mkdir -p logs

# Use env's python directly (most robust)
PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH

echo "== Downloading channels=$CHANNELS  range=$START → $END =="
echo "== Cadence: ALL daily rolling 8-day composites (use --anchor-only to subsample) =="
echo "== Resolution: stride-2 (0.025°) =="
echo

$PYBIN scripts/01_download.py \
    --channels "$CHANNELS" \
    --start "$START" \
    --end "$END" \
    --workers 8

echo
# Skip cube build if SKIP_CUBE=1 (e.g. running per-channel sibling jobs in parallel
# to avoid racing on the same cube.zarr write). Build once manually when all 4
# channels are done: sbatch hyak/build_cube.sbatch  OR run inline.
if [ "${SKIP_CUBE:-0}" = "1" ]; then
  echo "== SKIP_CUBE=1 — not building cube. Run build_cube.sbatch separately. =="
else
  echo "== Build cube =="
  $PYBIN scripts/02_build_cube.py
  echo
  echo "== Download + cube build complete =="
  echo "Cube location: $OAD_ROOT/data/cube.zarr"
  ls -lh data/cube.zarr 2>&1 | head -3
fi

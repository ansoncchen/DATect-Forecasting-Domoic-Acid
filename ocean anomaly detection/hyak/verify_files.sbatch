#!/bin/bash
#SBATCH --job-name=oad-verify
#SBATCH --partition=ckpt
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/oad-verify-%j.out
#SBATCH --error=logs/oad-verify-%j.err

set -euo pipefail

OAD_ROOT="/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid/ocean anomaly detection"
cd "$OAD_ROOT"

MIN_PER_CHANNEL="${MIN_PER_CHANNEL:-260}"   # 264 = 22 yr × 12 mo; allow 4-month slack for any partial Dec/Jan or genuine ERDDAP gaps

echo "=== Verifying all 4 channels have ≥ $MIN_PER_CHANNEL month-files ==="
ALL_OK=1
for ch in chla k490 nflh sst; do
  N=$(ls "data/raw/$ch/${ch}_"[0-9][0-9][0-9][0-9]_[0-9][0-9].nc 2>/dev/null | wc -l)
  STATUS="OK"
  if [ "$N" -lt "$MIN_PER_CHANNEL" ]; then
    STATUS="MISSING ($((MIN_PER_CHANNEL - N)) short)"
    ALL_OK=0
  fi
  echo "  $ch: $N/264  $STATUS"
done

if [ "$ALL_OK" = "1" ]; then
  echo "=== All channels complete — chain will continue. ==="
  exit 0
else
  echo "=== Incomplete — full pipeline will NOT auto-fire (afterok dependency). ==="
  exit 1
fi

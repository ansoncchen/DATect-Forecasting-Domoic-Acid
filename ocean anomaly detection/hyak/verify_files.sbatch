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

# Coverage threshold expressed as DISTINCT MONTHS-covered, not file count.
# This works regardless of whether files are per-month (chla_2003_07.nc)
# or per-day (chla_20030715.nc); we extract the YYYY-MM from any matching file.
MIN_MONTHS="${MIN_MONTHS:-258}"   # of 264 = 22yr × 12mo

PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python

echo "=== Verifying each of 4 channels covers ≥ $MIN_MONTHS distinct months ==="
ALL_OK=1
for ch in chla k490 nflh sst; do
  # Count distinct YYYY-MM across BOTH per-month and per-day filename patterns.
  N=$($PYBIN -c "
import os, re
have = set()
for f in os.listdir('data/raw/$ch'):
    # per-month: ${ch}_YYYY_MM.nc
    m = re.match(r'^${ch}_(\d{4})_(\d{2})\.nc\$', f)
    if m: have.add((m.group(1), m.group(2))); continue
    # per-day: ${ch}_YYYYMMDD.nc
    m = re.match(r'^${ch}_(\d{4})(\d{2})\d{2}\.nc\$', f)
    if m: have.add((m.group(1), m.group(2)))
print(len(have))
")
  STATUS="OK"
  if [ "$N" -lt "$MIN_MONTHS" ]; then
    STATUS="MISSING ($((MIN_MONTHS - N)) months short)"
    ALL_OK=0
  fi
  echo "  $ch: $N distinct months  $STATUS"
done

if [ "$ALL_OK" = "1" ]; then
  echo "=== All channels complete — chain will continue. ==="
  exit 0
else
  echo "=== Incomplete — full pipeline will NOT auto-fire. ==="
  exit 1
fi

#!/bin/bash
#SBATCH --job-name=oad-sanity
#SBATCH --partition=ckpt
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/oad-sanity-%j.out
#SBATCH --error=logs/oad-sanity-%j.err

set -euo pipefail

OAD_ROOT="${OAD_ROOT:-/gscratch/stf/ac283/oad_current}"  # must contain data/ and outputs/
PYBIN=${DATECT_PY:-/gscratch/stf/ac283/envs/datect_scratch/bin/python}
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH

cd "$OAD_ROOT" || { echo "OAD_ROOT not found"; exit 1; }
mkdir -p logs

echo "== Cloud-coverage confound check =="
$PYBIN scripts/07_cloud_confound.py
echo

echo "== Multi-step-ahead forecastability sweep (leads: 1, 7, 14, 28, 56 days) =="
$PYBIN scripts/08_multistep_forecast.py --leads 1 7 14 28 56

echo "== Done =="

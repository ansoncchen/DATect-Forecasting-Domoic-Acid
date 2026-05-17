#!/bin/bash
#SBATCH --job-name=oad-cube
#SBATCH --partition=ckpt
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/oad-cube-%j.out
#SBATCH --error=logs/oad-cube-%j.err

set -euo pipefail

# Standalone cube builder. Run after parallel-channel downloads finish:
#   sbatch hyak/build_cube.sbatch
#
# Or custom range/output:
#   sbatch --export=ALL,CUBE_ARGS="--year 2003 --out data/cube_2003.zarr" hyak/build_cube.sbatch

OAD_ROOT="/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid/ocean anomaly detection"
CUBE_ARGS="${CUBE_ARGS:-}"

cd "$OAD_ROOT"
mkdir -p logs

PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH

echo "== Building cube =="
echo "== Args: $CUBE_ARGS =="
$PYBIN scripts/02_build_cube.py $CUBE_ARGS
echo
echo "== Cube built =="
ls -lh data/cube*.zarr 2>&1 | head

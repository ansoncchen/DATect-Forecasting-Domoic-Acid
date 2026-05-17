#!/bin/bash
#SBATCH --job-name=oad-infer
#SBATCH --partition=gpu-2080ti       # stf has 2 free GPUs here; for big runs use --export=ALL,PARTITION=ckpt
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/oad-infer-%j.out
#SBATCH --error=logs/oad-infer-%j.err

set -euo pipefail


# Inference: runs all AE checkpoints + 2D baselines + 3D temporal baselines.
# Override:  sbatch --export=ALL,FLAGS="--baselines-only" hyak/infer.sbatch

OAD_ROOT="/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid/ocean anomaly detection"
FLAGS="${FLAGS:---all-ae --baselines-only --baselines-3d}"

cd "$OAD_ROOT" || { echo "OAD_ROOT not found"; exit 1; }
mkdir -p logs
# CUDA provided by conda env
# Use env's python directly (most robust)
PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH

echo "== FLAGS=$FLAGS =="
$PYBIN scripts/04_run_inference.py $FLAGS
echo "== Done =="

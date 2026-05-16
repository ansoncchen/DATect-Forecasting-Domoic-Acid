#!/bin/bash
#SBATCH --job-name=oad-infer
#SBATCH --partition=gpu-a40
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/oad-infer-%j.out
#SBATCH --error=logs/oad-infer-%j.err

# Inference: runs all AE checkpoints + 2D baselines + 3D temporal baselines.
# Override:  sbatch --export=ALL,FLAGS="--baselines-only" hyak/infer.sbatch

OAD_ROOT="/gscratch/stf/ac283/oad"
FLAGS="${FLAGS:---all-ae --baselines-only --baselines-3d}"

cd "$OAD_ROOT" || { echo "OAD_ROOT not found"; exit 1; }
mkdir -p logs
module load cuda/12.4 2>/dev/null || true
source .venv/bin/activate

echo "== FLAGS=$FLAGS =="
python scripts/04_run_inference.py $FLAGS
echo "== Done =="

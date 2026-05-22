#!/bin/bash
#SBATCH --job-name=oad-chain
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --array=0-3%1
# %1 = at most 1 concurrent task. Sequential because each chain mutates the
# shared data/processed/final_output.parquet — parallel writes corrupt it.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/oad-chain-%A_%a.out
#SBATCH --error=logs/oad-chain-%A_%a.err

set -euo pipefail

REPO_ROOT="/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid"
cd "$REPO_ROOT" || exit 1
mkdir -p logs chains/output

PYBIN=/gscratch/stf/ac283/envs/datect_scratch/bin/python
export PATH=/gscratch/stf/ac283/envs/datect_scratch/bin:$PATH

CHAINS=(lagged_pn beuti_derivatives nemo_mooring esp_offshore_pda ndbc_wind)
NAME="${CHAINS[$SLURM_ARRAY_TASK_ID]}"

echo "== Node: $(hostname) =="
echo "== Array task: $SLURM_ARRAY_TASK_ID  chain: $NAME =="
echo "== Started: $(date) =="

$PYBIN chains/run_chain.py --chain "$NAME"

echo "== Done: $(date) =="

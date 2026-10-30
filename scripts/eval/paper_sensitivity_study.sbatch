#!/bin/bash
#SBATCH --job-name=datect-sensitivity
#SBATCH --partition=ckpt
#SBATCH --account=stf-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=sensitivity-%j.out
#SBATCH --error=sensitivity-%j.err
#
# Paper appendix studies (seed 42, 20% sample unless noted). STUDY picks one:
#   features       feature-group ablation  -> results/ablation/paper_features_results.json
#   perturbations  design perturbations    -> results/ablation/paper_perturbations_results.json
#   classifier     spike-classifier gain importance over seeds 42-46
#                                          -> results/spike_classifier/importance_by_seed.csv
# One job per study; they write different files, so they can run at once:
#   sbatch --export=ALL,STUDY=features scripts/eval/paper_sensitivity_study.sbatch
# The two ablation studies skip finished experiments, so a ckpt requeue resumes.
set -euo pipefail

PY=${DATECT_PY:-/gscratch/stf/ac283/envs/datect_scratch/bin/python}
STUDY=${STUDY:?set STUDY=features, perturbations or classifier}
export TMPDIR=/gscratch/scrubbed/$USER/tmp
mkdir -p "$TMPDIR"

git rev-parse HEAD > "sensitivity_${STUDY}_sha.txt"
echo "== $STUDY @ $(date +%T) =="
if [ "$STUDY" = classifier ]; then
    $PY -u scripts/eval/spike_classifier_importance.py
else
    $PY -u scripts/eval/paper_sensitivity_study.py --study "$STUDY"
fi
echo "== done @ $(date +%T) =="

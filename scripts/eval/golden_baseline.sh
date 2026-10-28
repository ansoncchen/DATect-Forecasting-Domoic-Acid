#!/bin/bash
# Snapshot the headline evaluation outputs for a git ref (Hyak only).
#
# Creates a disposable worktree of <ref> under scrubbed storage and submits
# golden_baseline.sbatch inside it, so eval scripts can write their usual
# relative output paths without dirtying any working checkout.
#
# Usage (from any Hyak checkout of this repo):
#   bash scripts/eval/golden_baseline.sh <git-ref> [label]
# Then compare two runs:
#   python scripts/eval/compare_golden.py <golden_root>/<label_a> <golden_root>/<label_b>
set -euo pipefail

REF=${1:?usage: golden_baseline.sh <git-ref> [label]}
REPO=$(git rev-parse --show-toplevel)
SHA=$(git -C "$REPO" rev-parse --short "$REF")
LABEL=${2:-$SHA}
DEST=${GOLDEN_ROOT:-/gscratch/scrubbed/$USER/golden}/$LABEL

if [ -e "$DEST" ]; then
    echo "golden dir already exists: $DEST" >&2
    exit 1
fi
mkdir -p "$(dirname "$DEST")"
git -C "$REPO" worktree add --detach "$DEST" "$SHA"
sbatch --chdir="$DEST" "$DEST/scripts/eval/golden_baseline.sbatch"
echo "golden worktree: $DEST"

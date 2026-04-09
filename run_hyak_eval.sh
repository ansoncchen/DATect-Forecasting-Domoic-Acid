#!/usr/bin/env bash
# run_hyak_eval.sh
# Runs all paper evaluation steps sequentially on Hyak.
# Usage:
#   bash run_hyak_eval.sh           # all 4 steps + precompute cache
#   bash run_hyak_eval.sh --no-cache  # skip precompute_cache.py at the end

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
SKIP_CACHE=false
for arg in "$@"; do
    [[ "$arg" == "--no-cache" ]] && SKIP_CACHE=true
done

LOG_DIR="eval_results/hyak_run_logs"
mkdir -p "$LOG_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="$LOG_DIR/run_${RUN_ID}_summary.txt"

OMP_NUM_THREADS=1
export OMP_NUM_THREADS

# ── Helpers ──────────────────────────────────────────────────────────────────
step=0
total_steps=5
[[ "$SKIP_CACHE" == true ]] && total_steps=4

log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$SUMMARY_LOG"
}

run_step() {
    local label="$1"
    local logfile="$2"
    shift 2
    local cmd=("$@")

    step=$((step + 1))
    log "━━━ Step $step/$total_steps: $label ━━━"
    log "Command: ${cmd[*]}"
    log "Log: $logfile"

    local start_ts
    start_ts=$(date +%s)

    if "${cmd[@]}" > "$logfile" 2>&1; then
        local elapsed=$(( $(date +%s) - start_ts ))
        log "✓ PASSED in ${elapsed}s"
        # Pull the R² line from ralph_evaluate output if present
        grep -E "Pooled R2|pooled_r2|R\^2" "$logfile" | tail -5 >> "$SUMMARY_LOG" 2>/dev/null || true
    else
        local exit_code=$?
        local elapsed=$(( $(date +%s) - start_ts ))
        log "✗ FAILED (exit $exit_code) after ${elapsed}s — check $logfile"
        echo ""
        echo "Last 20 lines of $logfile:"
        tail -20 "$logfile"
        echo ""
        log "Aborting pipeline."
        exit "$exit_code"
    fi

    echo ""
}

# ── Pipeline ─────────────────────────────────────────────────────────────────
log "DATect Hyak Eval Pipeline — run $RUN_ID"
log "Steps: $total_steps | Skip cache: $SKIP_CACHE"
echo ""

run_step \
    "Standard eval (seed=123, 40%)" \
    "$LOG_DIR/run_${RUN_ID}_step1_ralph_standard.txt" \
    python3 ralph_evaluate.py --seed 123 --variant-name "blended-v2-seed123"

run_step \
    "Temporal holdout eval (2019-2023)" \
    "$LOG_DIR/run_${RUN_ID}_step2_ralph_temporal.txt" \
    python3 ralph_evaluate.py --seed 123 --temporal-holdout --variant-name "blended-v2-temporal-holdout"

run_step \
    "Bootstrap CIs — standard eval (paper Tables 1+2)" \
    "$LOG_DIR/run_${RUN_ID}_step3_metrics_standard.txt" \
    python3 eval_paper_metrics.py --seed 123

run_step \
    "Bootstrap CIs — temporal holdout (paper Table 4)" \
    "$LOG_DIR/run_${RUN_ID}_step4_metrics_temporal.txt" \
    python3 eval_paper_metrics.py --seed 123 --temporal-holdout

if [[ "$SKIP_CACHE" == false ]]; then
    run_step \
        "Precompute dashboard cache" \
        "$LOG_DIR/run_${RUN_ID}_step5_cache.txt" \
        python3 precompute_cache.py
fi

# ── Final summary ─────────────────────────────────────────────────────────────
log "━━━ All steps complete ━━━"
log "Summary written to: $SUMMARY_LOG"
echo ""
echo "Key output locations:"
echo "  eval_results/paper_metrics/        ← standard CI CSVs (Tables 1+2)"
echo "  eval_results/paper_metrics_temporal/ ← holdout CI CSVs (Table 4)"
echo "  $LOG_DIR/                          ← full logs per step"
echo ""
echo "Grep scoreboard R² from standard eval:"
grep -E "Pooled|WA mean|OR mean" \
    "$LOG_DIR/run_${RUN_ID}_step1_ralph_standard.txt" 2>/dev/null || true
echo ""
echo "Grep scoreboard R² from temporal holdout:"
grep -E "Pooled|WA mean|OR mean" \
    "$LOG_DIR/run_${RUN_ID}_step2_ralph_temporal.txt" 2>/dev/null || true

#!/usr/bin/env bash
# run_full_validation.sh
# Revalidates every number in the paper. Runs independent jobs in parallel.
#
# Usage:
#   nohup bash run_full_validation.sh > eval_results/hyak_run_logs/nohup_full_validation.txt 2>&1 & echo "PID: $!"
#
# Flags:
#   --no-cache       skip precompute_cache.py
#   --no-stability   skip scripts/eval/paper_stability_study.py (~3 hrs)
#   --quick          smoke test only (implies --no-stability)
#
# Output locations after completion:
#   eval_results/paper_metrics/           Tables 1+2 (standard CIs)
#   eval_results/paper_metrics_temporal/  Table 3 (temporal holdout CIs)
#   eval_results/stability/               Stability/seed tables
#   cache/                                Dashboard cache

set -uo pipefail
export OMP_NUM_THREADS=1

# ── Argument parsing ──────────────────────────────────────────────────────────
SKIP_CACHE=false
SKIP_STABILITY=false
QUICK=false
for arg in "$@"; do
    [[ "$arg" == "--no-cache"     ]] && SKIP_CACHE=true
    [[ "$arg" == "--no-stability" ]] && SKIP_STABILITY=true
    [[ "$arg" == "--quick"        ]] && QUICK=true && SKIP_STABILITY=true
done

# ── Setup ─────────────────────────────────────────────────────────────────────
LOG_DIR="eval_results/hyak_run_logs"
mkdir -p "$LOG_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="$LOG_DIR/full_validation_${RUN_ID}.txt"

log() {
    local msg="[$(date '+%Y-%m-%dT%H:%M:%S')] $*"
    echo "$msg" | tee -a "$SUMMARY_LOG"
}

log "════════════════════════════════════════════════════════"
log "  DATect Full Paper Validation — run $RUN_ID"
log "  skip_cache=$SKIP_CACHE  skip_stability=$SKIP_STABILITY  quick=$QUICK"
log "════════════════════════════════════════════════════════"
echo ""

# ── Background job tracker ────────────────────────────────────────────────────
declare -A JOB_PIDS
declare -A JOB_LOGS
declare -A JOB_LABELS

launch() {
    local key="$1"
    local label="$2"
    local logfile="$LOG_DIR/${RUN_ID}_${key}.txt"
    shift 2

    log "▶ LAUNCH [$key] $label"
    "$@" > "$logfile" 2>&1 &
    JOB_PIDS[$key]=$!
    JOB_LOGS[$key]="$logfile"
    JOB_LABELS[$key]="$label"
    log "  PID=${JOB_PIDS[$key]}  log=$logfile"
}

wait_for() {
    local key="$1"
    local pid="${JOB_PIDS[$key]}"
    local label="${JOB_LABELS[$key]}"
    local logfile="${JOB_LOGS[$key]}"

    log "⏳ WAITING [$key] $label (PID=$pid)..."
    local start_ts
    start_ts=$(date +%s)

    if wait "$pid"; then
        local elapsed=$(( $(date +%s) - start_ts ))
        log "✓ DONE [$key] ${elapsed}s"
        grep -E "Pooled R2|pooled_r2|R\^2|R2=|ensemble.*R2|transition_recall" \
            "$logfile" 2>/dev/null | tail -5 | while read -r line; do log "    $line"; done || true
    else
        local exit_code=$?
        local elapsed=$(( $(date +%s) - start_ts ))
        log "⚠ EXIT $exit_code [$key] ${elapsed}s — results may still be valid (gate failures are non-fatal)"
        tail -10 "$logfile" | while read -r line; do log "    $line"; done
    fi
    echo ""
}

# ── Group 1: Core regression metrics (parallel) ───────────────────────────────
log "── GROUP 1: Core regression metrics ──────────────────────"

launch "ci_standard" \
    "Tables 1+2 bootstrap CIs (eval_paper_metrics seed=123)" \
    python3 scripts/eval/eval_paper_metrics.py --seed 123 --force-rerun

launch "ci_temporal" \
    "Table 3 temporal holdout CIs (eval_paper_metrics --temporal-holdout)" \
    python3 scripts/eval/eval_paper_metrics.py --seed 123 --temporal-holdout --force-rerun

# ── Group 2: Spike detection (Table 9 + Fig 4) ────────────────────────────────
log "── GROUP 2: Spike detection ───────────────────────────────"

launch "spike_eval" \
    "Table 9 spike detection metrics" \
    python3 scripts/eval/spike_detection_eval.py

# ── Group 3: Ablation study (Appendix, dev set seed=42) ──────────────────────
log "── GROUP 3: Ablation study ────────────────────────────────"

if [[ "$QUICK" == true ]]; then
    log "  [QUICK] skipping paper_ablation_study.py"
else
    launch "ablation" \
        "Ablation table (paper_ablation_study.py, seed=42)" \
        python3 scripts/eval/paper_ablation_study.py
fi

# ── Group 4: Stability study (Appendix, ~3 hrs) ───────────────────────────────
log "── GROUP 4: Stability study ───────────────────────────────"

if [[ "$SKIP_STABILITY" == true ]]; then
    log "  [SKIPPED] paper_stability_study.py"
else
    launch "stability" \
        "Stability/seed tables (paper_stability_study.py, ~3 hrs)" \
        python3 scripts/eval/paper_stability_study.py
fi

# ── Group 5: Dashboard cache ──────────────────────────────────────────────────
log "── GROUP 5: Dashboard cache ───────────────────────────────"

if [[ "$SKIP_CACHE" == true ]]; then
    log "  [SKIPPED] precompute_cache.py"
else
    launch "cache" \
        "Dashboard cache (precompute_cache.py, ~45 min)" \
        python3 precompute_cache.py
fi

echo ""
log "All jobs launched. Waiting for completion..."
echo ""

# ── Wait for all jobs ─────────────────────────────────────────────────────────
wait_for "ci_standard"
wait_for "ci_temporal"
wait_for "spike_eval"
[[ "$QUICK" != true ]]        && wait_for "ablation"
[[ "$SKIP_STABILITY" != true ]] && wait_for "stability"
[[ "$SKIP_CACHE" != true ]]   && wait_for "cache"

# ── Post-stability: generate LaTeX tables ────────────────────────────────────
if [[ "$SKIP_STABILITY" != true ]]; then
    log "── POST-STABILITY: Generate LaTeX tables ──────────────────"
    python3 scripts/eval/paper_stability_table.py --latex \
        > "$LOG_DIR/${RUN_ID}_stability_latex.txt" 2>&1 \
        && log "✓ Stability LaTeX → $LOG_DIR/${RUN_ID}_stability_latex.txt" \
        || log "⚠ paper_stability_table.py failed — check log"
    echo ""
fi

# ── Final summary ─────────────────────────────────────────────────────────────
log "════════════════════════════════════════════════════════"
log "  ALL JOBS COMPLETE"
log "════════════════════════════════════════════════════════"
echo ""
log "Key outputs:"
log "  Tables 1+2 CIs : eval_results/paper_metrics/bootstrap_ci_summary.csv"
log "  Per-site CIs   : eval_results/paper_metrics/per_site_bootstrap_ci.csv"
log "  Table 3 CIs    : eval_results/paper_metrics_temporal/bootstrap_ci_summary.csv"
log "  Full logs      : $LOG_DIR/"
log "  Summary        : $SUMMARY_LOG"
echo ""

log "── Standard eval (Tables 1+2) ─────────────────────────────"
grep -E "TABLE 1|TABLE 2|Pooled|WA mean|OR mean|ensemble" \
    "$LOG_DIR/${RUN_ID}_ci_standard.txt" 2>/dev/null \
    | tail -20 | while read -r line; do log "  $line"; done || log "  (not found)"

log "── Temporal holdout scoreboard ────────────────────────────"
grep -E "Pooled|WA mean|OR mean" \
    "$LOG_DIR/${RUN_ID}_ci_temporal.txt" 2>/dev/null \
    | tail -8 | while read -r line; do log "  $line"; done || log "  (not found)"

log "── Bootstrap CI summary (standard) ───────────────────────"
[[ -f "eval_results/paper_metrics/bootstrap_ci_summary.csv" ]] \
    && cat "eval_results/paper_metrics/bootstrap_ci_summary.csv" \
         | while read -r line; do log "  $line"; done \
    || log "  (file not found)"

echo ""
log "Done. rsync cache to local when ready:"
log "  rsync -avz klone-login:/gscratch/stf/YOUR_UWNETID/DATect-Forecasting-Domoic-Acid/cache/ ./cache/"

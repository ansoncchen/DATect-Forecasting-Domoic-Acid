#!/usr/bin/env python3
"""
ralph_evaluate.py — Ralph Wiggum Loop Evaluation Harness
=========================================================

Runs a single config variant through the retrospective evaluation pipeline
and reports per-site R², spike metrics, and pass/fail gates against baselines.

Appends results to ralph_loop_log.jsonl for tracking across iterations.

Usage:
    python ralph_evaluate.py --variant-name "baseline" [--quick] [--seed 123]
    python ralph_evaluate.py --variant-name "huber_oregon" --quick
    python ralph_evaluate.py --variant-name "temporal-2019" --temporal-holdout
    python ralph_evaluate.py --variant-name "ceilings" --diagnostics
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config as cfg
from eval_paper_metrics import compute_spike_tables
from forecasting.raw_forecast_engine import RawForecastEngine
from forecasting.autocorrelation_diagnostic import (
    compute_autocorrelation_ceiling,
    print_ceiling_table,
)

WA_SITES = {"Kalaloch", "Quinault", "Copalis", "Twin Harbors", "Long Beach"}
OR_SITES = {"Clatsop Beach", "Cannon Beach", "Newport", "Coos Bay", "Gold Beach"}

BASELINES = {
    "Copalis":       {"r2": 0.789, "mae": 2.81},
    "Twin Harbors":  {"r2": 0.594, "mae": 4.66},
    "Quinault":      {"r2": 0.582, "mae": 4.13},
    "Long Beach":    {"r2": 0.631, "mae": 4.47},
    "Kalaloch":      {"r2": 0.480, "mae": 3.78},
    "Clatsop Beach": {"r2": 0.290, "mae": 6.16},
    "Cannon Beach":  {"r2": -0.044, "mae": 3.89},
    "Gold Beach":    {"r2": 0.041, "mae": 7.78},
    "Coos Bay":      {"r2": -0.039, "mae": 24.55},
    "Newport":       {"r2": -0.299, "mae": 10.53},
}

# Pooled baselines from paper Table 1 (seed=123, 40%, N=2181)
POOLED_BASELINES = {
    "ensemble": {"r2": 0.204, "mae": 6.42, "rmse": 17.92},
    "ridge":    {"r2": 0.202, "mae": 6.76, "rmse": 17.94},
    "naive":    {"r2": -0.426, "mae": 7.73, "rmse": 23.98},
}

SPIKE_BASELINES = {
    "transition_recall": 0.652,
    "spike_recall": 0.815,
    "spike_precision": 0.339,
    "spike_f2": 0.636,
}


# ---------------------------------------------------------------------------
# Per-site metrics
# ---------------------------------------------------------------------------

def compute_per_site_metrics(df: pd.DataFrame) -> dict:
    pred_col = "ensemble_prediction" if "ensemble_prediction" in df.columns else "predicted_da"

    has_naive = "naive_prediction" in df.columns
    has_ridge = "predicted_da_linear" in df.columns

    results = {}
    for site in cfg.SITES:
        site_df = df[df["site"] == site].dropna(subset=["actual_da", pred_col])
        n = len(site_df)
        if n < 3:
            results[site] = {
                "n": n, "r2": None, "mae": None, "rmse": None,
                "naive_r2": None, "ridge_r2": None,
                "beats_naive": None, "beats_ridge": None,
                "ml_earns_keep": None,
            }
            continue
        actual = site_df["actual_da"].values.astype(float)
        predicted = site_df[pred_col].values.astype(float)
        r2 = float(r2_score(actual, predicted))
        mae = float(mean_absolute_error(actual, predicted))
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))

        naive_r2 = None
        if has_naive:
            mask = site_df["naive_prediction"].notna()
            if mask.sum() >= 3:
                naive_r2 = round(float(r2_score(
                    site_df.loc[mask, "actual_da"].values.astype(float),
                    site_df.loc[mask, "naive_prediction"].values.astype(float),
                )), 4)

        ridge_r2 = None
        if has_ridge:
            mask = site_df["predicted_da_linear"].notna()
            if mask.sum() >= 3:
                ridge_r2 = round(float(r2_score(
                    site_df.loc[mask, "actual_da"].values.astype(float),
                    site_df.loc[mask, "predicted_da_linear"].values.astype(float),
                )), 4)

        beats_naive = (r2 > naive_r2) if naive_r2 is not None else None
        beats_ridge = (r2 > ridge_r2 - 0.01) if ridge_r2 is not None else None
        ml_earns_keep = None
        if beats_naive is not None and beats_ridge is not None:
            ml_earns_keep = bool(beats_naive and beats_ridge)

        results[site] = {
            "n": n, "r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2),
            "naive_r2": naive_r2, "ridge_r2": ridge_r2,
            "beats_naive": beats_naive, "beats_ridge": beats_ridge,
            "ml_earns_keep": ml_earns_keep,
        }
    return results


def compute_pooled_metrics(df: pd.DataFrame) -> dict:
    pred_col = "ensemble_prediction" if "ensemble_prediction" in df.columns else "predicted_da"
    df_clean = df.dropna(subset=["actual_da", pred_col])
    actual = df_clean["actual_da"].values.astype(float)
    predicted = df_clean[pred_col].values.astype(float)
    return {
        "n": len(df_clean),
        "r2": round(float(r2_score(actual, predicted)), 4),
        "mae": round(float(mean_absolute_error(actual, predicted)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2),
    }


def compute_tier_metrics(per_site: dict) -> dict:
    wa_r2 = [m["r2"] for s, m in per_site.items() if s in WA_SITES and m["r2"] is not None]
    or_r2 = [m["r2"] for s, m in per_site.items() if s in OR_SITES and m["r2"] is not None]
    return {
        "wa_mean_r2": round(float(np.mean(wa_r2)), 4) if wa_r2 else None,
        "or_mean_r2": round(float(np.mean(or_r2)), 4) if or_r2 else None,
        "wa_sites_above_04": sum(1 for r in wa_r2 if r >= 0.4),
        "or_sites_above_04": sum(1 for r in or_r2 if r >= 0.4),
        "total_sites_above_04": sum(1 for r in wa_r2 + or_r2 if r >= 0.4),
    }


# ---------------------------------------------------------------------------
# Diagnostic: bias near threshold + episodic error decomposition
# ---------------------------------------------------------------------------

def compute_bias_analysis(df: pd.DataFrame) -> dict:
    """
    Compute systematic under-prediction bias near the spike threshold.

    Focuses on the 'transition zone' where actual DA is between 12 and 30 µg/g
    — the range where correct prediction matters most for alerts.
    """
    pred_col = "ensemble_prediction" if "ensemble_prediction" in df.columns else "predicted_da"
    threshold = getattr(cfg, "SPIKE_THRESHOLD", 20.0)
    lo, hi = threshold * 0.6, threshold * 1.5  # ~12-30 µg/g for threshold=20

    df_clean = df.dropna(subset=["actual_da", pred_col]).copy()
    df_clean["actual_da"] = df_clean["actual_da"].astype(float)
    df_clean[pred_col] = df_clean[pred_col].astype(float)

    transition_mask = (df_clean["actual_da"] >= lo) & (df_clean["actual_da"] <= hi)
    transition_df = df_clean[transition_mask]

    result = {"n_transition": int(transition_mask.sum())}

    if len(transition_df) >= 3:
        act = transition_df["actual_da"].values
        pred = transition_df[pred_col].values
        bias = float(np.mean(act - pred))  # positive = under-prediction
        result["transition_bias"] = round(bias, 2)
        result["transition_mean_actual"] = round(float(np.mean(act)), 2)
        result["transition_mean_predicted"] = round(float(np.mean(pred)), 2)

        # Per-site breakdown
        site_bias = {}
        for site in cfg.SITES:
            s_df = transition_df[transition_df["site"] == site]
            if len(s_df) >= 2:
                site_bias[site] = round(float(np.mean(
                    s_df["actual_da"].values - s_df[pred_col].values
                )), 2)
        result["site_bias"] = site_bias
    else:
        result["transition_bias"] = None
        result["transition_mean_actual"] = None
        result["transition_mean_predicted"] = None
        result["site_bias"] = {}

    return result


def compute_episodic_error(df: pd.DataFrame) -> dict:
    """
    Decompose error into 'episodic' (top 5% worst weeks) vs 'chronic'.

    If 80% of MSE comes from top 5% of weeks, the problem is extreme-event
    prediction, not general model weakness.
    """
    pred_col = "ensemble_prediction" if "ensemble_prediction" in df.columns else "predicted_da"
    df_clean = df.dropna(subset=["actual_da", pred_col]).copy()
    df_clean["sq_err"] = (
        df_clean["actual_da"].astype(float) - df_clean[pred_col].astype(float)
    ) ** 2

    total_mse = float(df_clean["sq_err"].sum())
    n = len(df_clean)
    top_n = max(1, int(np.ceil(0.05 * n)))
    worst = df_clean.nlargest(top_n, "sq_err")

    top5_mse = float(worst["sq_err"].sum())
    frac = top5_mse / total_mse if total_mse > 0 else 0.0

    result = {
        "n_total": n,
        "n_top5pct": top_n,
        "mse_top5pct_fraction": round(frac, 4),
        "worst_rows": [],
    }

    # Top 5 worst rows
    for _, row in worst.head(5).iterrows():
        entry = {
            "site": str(row.get("site", "?")),
            "abs_err": round(float(row["sq_err"] ** 0.5), 1),
        }
        if "date" in row:
            entry["date"] = str(row["date"])[:10]
        if "actual_da" in row:
            entry["actual"] = round(float(row["actual_da"]), 1)
        if pred_col in row:
            entry["predicted"] = round(float(row[pred_col]), 1)
        result["worst_rows"].append(entry)

    # Per-site: fraction of site's MSE in top 5% worst weeks
    site_episodic = {}
    for site in cfg.SITES:
        s_df = df_clean[df_clean["site"] == site]
        if len(s_df) < 5:
            continue
        s_top_n = max(1, int(np.ceil(0.05 * len(s_df))))
        s_worst = s_df.nlargest(s_top_n, "sq_err")
        s_total = float(s_df["sq_err"].sum())
        if s_total > 0:
            site_episodic[site] = round(float(s_worst["sq_err"].sum()) / s_total, 4)
    result["site_mse_top5pct_fraction"] = site_episodic

    return result


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def apply_gates(per_site: dict, spike_metrics: dict) -> dict:
    gates = {"r2_pass": True, "spike_pass": True, "baseline_pass": True, "reasons": []}

    sites_losing_to_ridge = []
    sites_losing_to_naive = []

    for site, metrics in per_site.items():
        if metrics["r2"] is None:
            continue
        baseline_r2 = BASELINES.get(site, {}).get("r2", 0)
        delta = metrics["r2"] - baseline_r2

        if site in WA_SITES and metrics["r2"] < 0.40:
            gates["r2_pass"] = False
            gates["reasons"].append(f"WA site {site} dropped below 0.40: {metrics['r2']:.3f}")

        if delta < -0.03:
            gates["reasons"].append(f"{site} regressed by {delta:+.3f}")

        if metrics.get("beats_naive") is False:
            sites_losing_to_naive.append(site)
        if metrics.get("beats_ridge") is False:
            sites_losing_to_ridge.append(site)

    if sites_losing_to_naive:
        gates["baseline_pass"] = False
        gates["reasons"].append(
            f"ML loses to naive at: {', '.join(sites_losing_to_naive)}"
        )
    if sites_losing_to_ridge:
        gates["reasons"].append(
            f"ML loses to Ridge at: {', '.join(sites_losing_to_ridge)} (consider fallback)"
        )

    if spike_metrics.get("transition_recall", 0) < 0.50:
        gates["spike_pass"] = False
        gates["reasons"].append(
            f"Transition recall {spike_metrics.get('transition_recall', 0):.3f} < 0.50"
        )

    gates["overall_pass"] = gates["r2_pass"] and gates["spike_pass"]
    return gates


# ---------------------------------------------------------------------------
# Scoreboard printer
# ---------------------------------------------------------------------------

def print_scoreboard(
    per_site: dict,
    tier: dict,
    pooled: dict,
    spike: dict,
    gates: dict,
    bias: dict | None = None,
    episodic: dict | None = None,
):
    print("\n" + "=" * 120)
    print("RALPH WIGGUM SCOREBOARD")
    print("=" * 120)

    has_baselines = any(m.get("naive_r2") is not None or m.get("ridge_r2") is not None
                        for m in per_site.values())

    if has_baselines:
        header = (f"  {'Site':<18s}  {'N':>4s}  {'ML R²':>7s}  {'Ridge':>7s}  {'Naive':>7s}  "
                  f"{'vs Base':>7s}  {'MAE':>7s}  {'Beats?':>12s}  {'Status':>10s}")
    else:
        header = (f"  {'Site':<18s}  {'N':>4s}  {'R²':>7s}  {'Baseline':>9s}  "
                  f"{'Delta':>7s}  {'MAE':>7s}  {'Status':>10s}")
    print(header)
    print("  " + "-" * 116)

    for site in cfg.SITES:
        m = per_site.get(site, {})
        if m.get("r2") is None:
            print(f"  {site:<18s}  {'n/a':>4s}  — no data —")
            continue
        base_r2 = BASELINES.get(site, {}).get("r2", 0)
        delta = m["r2"] - base_r2
        status = "DONE" if m["r2"] >= 0.4 else "NEEDS WORK" if m["r2"] >= 0.1 else "CRITICAL"

        if has_baselines:
            naive_str = f"{m['naive_r2']:>7.4f}" if m.get("naive_r2") is not None else "    n/a"
            ridge_str = f"{m['ridge_r2']:>7.4f}" if m.get("ridge_r2") is not None else "    n/a"
            beat_parts = []
            if m.get("beats_naive") is True:
                beat_parts.append("N")
            elif m.get("beats_naive") is False:
                beat_parts.append("!N")
            if m.get("beats_ridge") is True:
                beat_parts.append("R")
            elif m.get("beats_ridge") is False:
                beat_parts.append("!R")
            beats_str = ",".join(beat_parts) if beat_parts else "?"
            print(f"  {site:<18s}  {m['n']:>4d}  {m['r2']:>7.4f}  {ridge_str}  {naive_str}  "
                  f"{delta:>+7.4f}  {m['mae']:>7.2f}  {beats_str:>12s}  {status:>10s}")
        else:
            print(f"  {site:<18s}  {m['n']:>4d}  {m['r2']:>7.4f}  {base_r2:>9.4f}  "
                  f"{delta:>+7.4f}  {m['mae']:>7.2f}  {status:>10s}")

    print(f"\n  Pooled: R²={pooled['r2']:.4f}  MAE={pooled['mae']:.2f}  N={pooled['n']}")
    print(f"  Paper baselines — Ridge: {POOLED_BASELINES['ridge']['r2']:.3f}  "
          f"Naive: {POOLED_BASELINES['naive']['r2']:.3f}  "
          f"Prev ensemble: {POOLED_BASELINES['ensemble']['r2']:.3f}")
    if tier.get("wa_mean_r2") is not None and tier.get("or_mean_r2") is not None:
        print(f"  WA mean R²: {tier['wa_mean_r2']:.4f}  |  OR mean R²: {tier['or_mean_r2']:.4f}")
    print(f"  Sites >= 0.4: {tier['total_sites_above_04']}/10 "
          f"(WA: {tier['wa_sites_above_04']}/5, OR: {tier['or_sites_above_04']}/5)")

    # Baseline comparison summary
    n_beating_naive = sum(1 for m in per_site.values() if m.get("beats_naive") is True)
    n_beating_ridge = sum(1 for m in per_site.values() if m.get("beats_ridge") is True)
    n_tested_naive = sum(1 for m in per_site.values() if m.get("beats_naive") is not None)
    n_tested_ridge = sum(1 for m in per_site.values() if m.get("beats_ridge") is not None)
    if n_tested_naive > 0:
        print(f"\n  ML beats naive: {n_beating_naive}/{n_tested_naive} sites")
    if n_tested_ridge > 0:
        print(f"  ML beats Ridge: {n_beating_ridge}/{n_tested_ridge} sites")
    n_ek = sum(1 for m in per_site.values() if m.get("ml_earns_keep") is True)
    n_ek_den = sum(1 for m in per_site.values() if m.get("ml_earns_keep") is not None)
    if n_ek_den > 0:
        print(f"  ML earns keep (beats Ridge & naive): {n_ek}/{n_ek_den} sites")

    print(f"\n  Spike — Transition recall: {spike.get('transition_recall', 'n/a')}")
    print(f"  Spike — Event recall: {spike.get('event_recall', 'n/a')}")
    print(f"  Spike — Hybrid FP rate: {spike.get('hybrid_fp_rate', 'n/a')}")

    # Bias/variance diagnostics
    if bias and bias.get("transition_bias") is not None:
        print(f"\n  Bias (transition zone {getattr(cfg, 'SPIKE_THRESHOLD', 20)*0.6:.0f}"
              f"-{getattr(cfg, 'SPIKE_THRESHOLD', 20)*1.5:.0f} µg/g, N={bias['n_transition']}):")
        print(f"    Mean actual: {bias['transition_mean_actual']:.1f}  "
              f"Mean predicted: {bias['transition_mean_predicted']:.1f}  "
              f"Under-prediction: {bias['transition_bias']:+.1f} µg/g")
        if bias.get("site_bias"):
            worst_bias = sorted(bias["site_bias"].items(), key=lambda x: -abs(x[1]))[:3]
            print(f"    Top biased sites: " +
                  ", ".join(f"{s}: {v:+.1f}" for s, v in worst_bias))

    # Episodic error diagnostics
    if episodic and episodic.get("mse_top5pct_fraction") is not None:
        print(f"\n  Error decomposition (N={episodic['n_total']}):")
        print(f"    Top 5% worst weeks ({episodic['n_top5pct']} rows) account for "
              f"{episodic['mse_top5pct_fraction']:.1%} of total MSE")
        if episodic.get("worst_rows"):
            print("    5 worst predictions:")
            for row in episodic["worst_rows"]:
                print(f"      {row.get('site','?')} {row.get('date','?')}: "
                      f"actual={row.get('actual','?'):.1f}  "
                      f"pred={row.get('predicted','?'):.1f}  "
                      f"err={row.get('abs_err','?'):.1f}")

    verdict = "PASS" if gates["overall_pass"] else "FAIL"
    print(f"\n  Gate verdict: {verdict}")
    if gates["reasons"]:
        for r in gates["reasons"]:
            print(f"    - {r}")
    print("=" * 120)


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    variant_name: str,
    seed: int = 123,
    sample_fraction: float = 0.40,
    enable_parallel: bool = True,
    temporal_holdout: bool = False,
) -> dict:
    t0 = time.time()

    orig_seed = cfg.RANDOM_SEED
    orig_frac = cfg.TEST_SAMPLE_FRACTION
    orig_parallel = cfg.ENABLE_PARALLEL
    orig_min_test_date = getattr(cfg, "MIN_TEST_DATE", "2003-01-01")

    cfg.RANDOM_SEED = seed
    cfg.TEST_SAMPLE_FRACTION = sample_fraction
    cfg.ENABLE_PARALLEL = enable_parallel
    os.environ["DATECT_RANDOM_SEED"] = str(seed)
    os.environ["DATECT_TEST_SAMPLE_FRACTION"] = str(sample_fraction)
    os.environ["DATECT_ENABLE_PARALLEL"] = "true" if enable_parallel else "false"

    if temporal_holdout:
        cutoff = getattr(cfg, "TEMPORAL_HOLDOUT_CUTOFF", "2019-01-01")
        cfg.MIN_TEST_DATE = cutoff
        # Use all post-cutoff measurements (fraction=1.0 effectively)
        cfg.TEST_SAMPLE_FRACTION = getattr(cfg, "TEMPORAL_HOLDOUT_FRACTION", 1.0)
        print(f"  [temporal-holdout] Using test dates >= {cutoff}, fraction={cfg.TEST_SAMPLE_FRACTION}")

    try:
        engine = RawForecastEngine(validate_on_init=False)
        engine.random_seed = seed
        df = engine.run_retrospective_evaluation(
            task="regression", model_type="ensemble",
        )
    finally:
        cfg.RANDOM_SEED = orig_seed
        cfg.TEST_SAMPLE_FRACTION = orig_frac
        cfg.ENABLE_PARALLEL = orig_parallel
        cfg.MIN_TEST_DATE = orig_min_test_date
        os.environ.pop("DATECT_RANDOM_SEED", None)
        os.environ.pop("DATECT_TEST_SAMPLE_FRACTION", None)
        os.environ.pop("DATECT_ENABLE_PARALLEL", None)

    elapsed = time.time() - t0

    if df is None or df.empty:
        return {"variant": variant_name, "status": "failed", "elapsed_s": round(elapsed, 1)}

    # Normalise column names
    if "actual_da_raw" in df.columns and "actual_da" not in df.columns:
        df = df.rename(columns={"actual_da_raw": "actual_da"})
    if "ensemble_prediction" not in df.columns and "predicted_da" in df.columns:
        df["ensemble_prediction"] = df["predicted_da"]

    per_site = compute_per_site_metrics(df)
    pooled = compute_pooled_metrics(df)
    tier = compute_tier_metrics(per_site)
    bias = compute_bias_analysis(df)
    episodic = compute_episodic_error(df)

    spike_metrics = {}
    try:
        event_df, transition_df = compute_spike_tables(
            df,
            spike_threshold=getattr(cfg, "SPIKE_THRESHOLD", 20.0),
            reg_alert_threshold=getattr(cfg, "SPIKE_REGRESSION_ALERT_THRESHOLD", 12.0),
        )
        for _, row in transition_df.iterrows():
            if row["model"] == "classifier":
                spike_metrics["transition_recall"] = round(float(row["recall"]), 4)
                spike_metrics["transition_f1"] = round(float(row["f1"]), 4)
            if row["model"] == "hybrid":
                spike_metrics["hybrid_transition_recall"] = round(float(row["recall"]), 4)
        for _, row in event_df.iterrows():
            if row["model"] == "classifier":
                spike_metrics["event_recall"] = round(float(row["recall"]), 4)
                spike_metrics["event_precision"] = round(float(row["precision"]), 4)
            if row["model"] == "hybrid":
                spike_metrics["hybrid_event_recall"] = round(float(row["recall"]), 4)
                spike_metrics["hybrid_fp_rate"] = (
                    round(1.0 - float(row["precision"]), 4) if row["precision"] > 0 else None
                )
    except Exception as exc:
        spike_metrics["error"] = str(exc)

    gates = apply_gates(per_site, spike_metrics)
    print_scoreboard(per_site, tier, pooled, spike_metrics, gates, bias=bias, episodic=episodic)

    result = {
        "variant": variant_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "seed": seed,
        "sample_fraction": sample_fraction,
        "temporal_holdout": temporal_holdout,
        "elapsed_s": round(elapsed, 1),
        "pooled": pooled,
        "tier": tier,
        "per_site": per_site,
        "spike": spike_metrics,
        "bias": bias,
        "episodic": {k: v for k, v in episodic.items() if k != "worst_rows"},
        "gates": gates,
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ralph Wiggum Loop Evaluator")
    parser.add_argument("--variant-name", required=True, help="Name for this variant/experiment")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sample-fraction", type=float, default=0.40)
    parser.add_argument("--quick", action="store_true", help="Use 10%% sample for fast iteration")
    parser.add_argument("--temporal-holdout", action="store_true",
                        help="Evaluate only on post-TEMPORAL_HOLDOUT_CUTOFF dates (honest out-of-sample)")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Run and print the per-site autocorrelation ceiling table")
    parser.add_argument("--disable-parallel", action="store_true")
    parser.add_argument("--log-file", default="ralph_loop_log.jsonl")
    args = parser.parse_args()

    fraction = 0.10 if args.quick else args.sample_fraction

    if args.diagnostics:
        print("\nRunning per-site autocorrelation ceiling diagnostic...")
        from forecasting.raw_data_forecaster import load_raw_da_measurements
        raw = load_raw_da_measurements()
        results = compute_autocorrelation_ceiling(raw)
        print_ceiling_table(results)
        print()

    print(f"\n{'='*60}")
    print(f"Ralph Wiggum Loop — Evaluating: {args.variant_name}")
    print(f"  seed={args.seed}  fraction={fraction}  "
          f"temporal_holdout={args.temporal_holdout}  "
          f"parallel={not args.disable_parallel}")
    print(f"{'='*60}\n")

    result = run_evaluation(
        variant_name=args.variant_name,
        seed=args.seed,
        sample_fraction=fraction,
        enable_parallel=not args.disable_parallel,
        temporal_holdout=args.temporal_holdout,
    )

    with open(args.log_file, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")
    print(f"\nLogged to {args.log_file}")

    if result.get("status") == "ok":
        verdict = "PASS" if result["gates"]["overall_pass"] else "FAIL"
        print(f"\nFinal verdict: {verdict}")
        return 0 if result["gates"]["overall_pass"] else 1
    else:
        print(f"\nEvaluation failed: {result}")
        return 2


if __name__ == "__main__":
    exit(main())

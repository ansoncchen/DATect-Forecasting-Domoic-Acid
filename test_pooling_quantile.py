#!/usr/bin/env python3
"""
A/B test: Cross-site pooling + quantile regression vs baseline.

Runs retrospective evaluation under three configurations:
  A) Baseline       — current production (no pooling, MSE objective)
  B) Pooling only   — cross-site pooled training for struggle sites
  C) Pooling + Q70  — pooling + quantile regression (alpha=0.7) on XGB

Reports per-site R², MAE, RMSE, spike recall, and classification accuracy.

Usage (Hyak):
    python test_pooling_quantile.py                   # full run, all sites
    python test_pooling_quantile.py --sites Newport "Cannon Beach"  # subset
    python test_pooling_quantile.py --n-anchors 100   # fewer test points (faster)
    python test_pooling_quantile.py --quick            # fast smoke test (20 anchors)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import config
from forecasting.raw_forecast_engine import RawForecastEngine
from forecasting.raw_data_forecaster import (
    load_raw_da_measurements,
    aggregate_raw_to_weekly,
    build_raw_feature_frame,
)
from forecasting.per_site_models import (
    get_site_ensemble_weights,
    SITE_SPECIFIC_CONFIGS,
)

warnings.filterwarnings("ignore")

SPIKE_THRESHOLD = config.SPIKE_THRESHOLD  # 20.0 µg/g


# ---------------------------------------------------------------------------
# Test-point sampling (mirrors run_retrospective_evaluation logic)
# ---------------------------------------------------------------------------

def sample_test_points(
    engine: RawForecastEngine,
    sites: list[str] | None = None,
    n_per_site: int | None = None,
    seed: int = 123,
) -> pd.DataFrame:
    """Sample leak-free test points from raw DA measurements."""
    feature_frame = engine._load_feature_frame()
    raw_data = engine._raw_data_cache

    min_test_ts = pd.Timestamp(getattr(config, "MIN_TEST_DATE", "2008-01-01"))
    min_training = engine.min_training_samples
    forecast_horizon = config.FORECAST_HORIZON_DAYS
    history_frac = getattr(config, "HISTORY_REQUIREMENT_FRACTION", 0.33)

    candidate_raw = raw_data[raw_data["date"] >= min_test_ts].copy()
    if sites:
        candidate_raw = candidate_raw[candidate_raw["site"].isin(sites)]

    site_total_counts = raw_data.groupby("site")["date"].size().to_dict()

    # Filter for sufficient history
    valid_rows = []
    for _, row in candidate_raw.iterrows():
        anchor_dt = row["date"] - pd.Timedelta(days=forecast_horizon)
        site = row["site"]
        total_site = site_total_counts.get(site, 0)
        if total_site == 0:
            continue
        min_required = max(int(np.ceil(history_frac * total_site)), min_training)
        n_history = len(
            raw_data[(raw_data["site"] == site) & (raw_data["date"] <= anchor_dt)]
        )
        if n_history < min_required:
            continue
        site_history = feature_frame[
            (feature_frame["site"] == site)
            & (feature_frame["date"] <= anchor_dt)
            & (feature_frame["da_raw"].notna())
        ]
        if len(site_history) >= min_training:
            valid_rows.append(row)

    if not valid_rows:
        print("ERROR: No valid test samples found")
        sys.exit(1)

    valid_df = pd.DataFrame(valid_rows)

    # Per-site sampling
    rng = np.random.RandomState(seed)
    sampled = []
    for site, site_df in valid_df.groupby("site"):
        site_df = site_df.sort_values("date")
        n_candidates = len(site_df)
        total_site = site_total_counts.get(site, n_candidates)
        test_frac = getattr(config, "TEST_SAMPLE_FRACTION", 0.20)
        target = min(int(np.ceil(test_frac * total_site)), n_candidates)
        if n_per_site is not None:
            target = min(target, n_per_site)
        if target <= 0:
            continue
        idx = rng.choice(n_candidates, size=target, replace=False)
        sampled.append(site_df.iloc[idx])

    test_points = pd.concat(sampled, ignore_index=True)
    test_points = test_points.sort_values(["site", "date"]).drop_duplicates(
        ["date", "site"]
    )
    return test_points


# ---------------------------------------------------------------------------
# Run forecasts under a specific config
# ---------------------------------------------------------------------------

def run_experiment(
    engine: RawForecastEngine,
    test_points: pd.DataFrame,
    label: str,
    use_pooled: bool,
    quantile_alpha: float | None,
) -> pd.DataFrame:
    """Run all test points through generate_single_forecast."""
    # Temporarily patch config
    original_pooled = {}
    original_q_alpha = getattr(config, "QUANTILE_REGRESSION_ALPHA", None)

    if not use_pooled:
        # Disable pooling for all sites
        for site_name, cfg in SITE_SPECIFIC_CONFIGS.items():
            original_pooled[site_name] = cfg.get("use_pooled_training", False)
            cfg["use_pooled_training"] = False

    config.QUANTILE_REGRESSION_ALPHA = quantile_alpha

    results = []
    n_total = len(test_points)
    t0 = time.time()

    for i, (_, row) in enumerate(test_points.iterrows()):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_total - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{label}] {i+1}/{n_total} "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                flush=True,
            )

        try:
            result = engine.generate_single_forecast(
                data_path=config.FINAL_OUTPUT_PATH,
                forecast_date=row["date"],
                site=row["site"],
                task="regression",
                model_type="ensemble",
            )
            if result is not None:
                results.append({
                    "date": row["date"],
                    "site": row["site"],
                    "actual_da": row["da_raw"],
                    "predicted_da": result["predicted_da"],
                    "xgb_prediction": result.get("xgb_prediction"),
                    "rf_prediction": result.get("rf_prediction"),
                    "naive_prediction": result.get("naive_prediction"),
                    "ensemble_prediction": result.get("ensemble_prediction"),
                    "training_samples": result.get("training_samples"),
                    "pooled_training": result.get("pooled_training", False),
                })
        except Exception as e:
            print(f"  WARNING: {row['site']} {row['date'].date()} failed: {e}")
            continue

    # Restore config
    for site_name, orig_val in original_pooled.items():
        SITE_SPECIFIC_CONFIGS[site_name]["use_pooled_training"] = orig_val
    config.QUANTILE_REGRESSION_ALPHA = original_q_alpha

    elapsed = time.time() - t0
    print(f"  [{label}] Done: {len(results)}/{n_total} succeeded in {elapsed:.0f}s")
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, label: str) -> dict:
    """Compute per-site and aggregate metrics."""
    metrics = {"label": label, "sites": {}, "aggregate": {}}

    valid = df.dropna(subset=["actual_da", "predicted_da"])
    if valid.empty:
        return metrics

    # Aggregate
    metrics["aggregate"] = _site_metrics(valid)

    # Per-site
    for site, site_df in valid.groupby("site"):
        if len(site_df) >= 2:
            metrics["sites"][site] = _site_metrics(site_df)

    return metrics


def _site_metrics(df: pd.DataFrame) -> dict:
    """Compute metrics for a single site or aggregate."""
    actual = df["actual_da"].values
    predicted = df["predicted_da"].values

    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # Spike detection
    actual_spike = actual >= SPIKE_THRESHOLD
    predicted_spike = predicted >= SPIKE_THRESHOLD
    n_actual_spikes = actual_spike.sum()

    if n_actual_spikes > 0:
        spike_recall = (actual_spike & predicted_spike).sum() / n_actual_spikes
    else:
        spike_recall = float("nan")

    n_predicted_spikes = predicted_spike.sum()
    if n_predicted_spikes > 0:
        spike_precision = (actual_spike & predicted_spike).sum() / n_predicted_spikes
    else:
        spike_precision = float("nan")

    # Classification accuracy (4 categories)
    def categorize(vals):
        cats = np.zeros(len(vals), dtype=int)
        cats[vals >= 5] = 1
        cats[vals >= 20] = 2
        cats[vals >= 40] = 3
        return cats

    actual_cat = categorize(actual)
    pred_cat = categorize(predicted)
    accuracy = (actual_cat == pred_cat).mean()

    return {
        "n": len(df),
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "spike_recall": round(spike_recall, 4) if not np.isnan(spike_recall) else None,
        "spike_precision": round(spike_precision, 4) if not np.isnan(spike_precision) else None,
        "n_actual_spikes": int(n_actual_spikes),
        "n_predicted_spikes": int(n_predicted_spikes),
        "category_accuracy": round(accuracy, 4),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison(all_metrics: list[dict]):
    """Print a formatted comparison table."""

    # Aggregate comparison
    print("\n" + "=" * 90)
    print("AGGREGATE RESULTS")
    print("=" * 90)
    header = f"{'Config':<25} {'N':>5} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'Spike Rec':>10} {'Spike Prec':>11} {'Cat Acc':>8}"
    print(header)
    print("-" * 90)
    for m in all_metrics:
        agg = m["aggregate"]
        if not agg:
            continue
        print(
            f"{m['label']:<25} {agg['n']:>5} {agg['r2']:>8.4f} {agg['mae']:>8.2f} "
            f"{agg['rmse']:>8.2f} {_fmt(agg['spike_recall']):>10} "
            f"{_fmt(agg['spike_precision']):>11} {agg['category_accuracy']:>8.4f}"
        )

    # Per-site comparison
    all_sites = sorted(
        set().union(*(m["sites"].keys() for m in all_metrics))
    )

    print("\n" + "=" * 90)
    print("PER-SITE R² COMPARISON")
    print("=" * 90)
    header = f"{'Site':<20}"
    for m in all_metrics:
        header += f" {m['label']:>20}"
    header += f" {'Best':>20}"
    print(header)
    print("-" * (20 + 21 * (len(all_metrics) + 1)))

    for site in all_sites:
        row = f"{site:<20}"
        values = []
        for m in all_metrics:
            if site in m["sites"]:
                r2 = m["sites"][site]["r2"]
                values.append((r2, m["label"]))
                row += f" {r2:>20.4f}"
            else:
                values.append((float("-inf"), m["label"]))
                row += f" {'N/A':>20}"
        best_label = max(values, key=lambda x: x[0])[1]
        row += f" {best_label:>20}"
        print(row)

    # Per-site MAE comparison
    print("\n" + "=" * 90)
    print("PER-SITE MAE COMPARISON")
    print("=" * 90)
    header = f"{'Site':<20}"
    for m in all_metrics:
        header += f" {m['label']:>20}"
    header += f" {'Best':>20}"
    print(header)
    print("-" * (20 + 21 * (len(all_metrics) + 1)))

    for site in all_sites:
        row = f"{site:<20}"
        values = []
        for m in all_metrics:
            if site in m["sites"]:
                mae = m["sites"][site]["mae"]
                values.append((mae, m["label"]))
                row += f" {mae:>20.2f}"
            else:
                values.append((float("inf"), m["label"]))
                row += f" {'N/A':>20}"
        best_label = min(values, key=lambda x: x[0])[1]
        row += f" {best_label:>20}"
        print(row)

    # Per-site spike recall comparison
    print("\n" + "=" * 90)
    print("PER-SITE SPIKE RECALL COMPARISON")
    print("=" * 90)
    header = f"{'Site':<20} {'Spikes':>7}"
    for m in all_metrics:
        header += f" {m['label']:>20}"
    print(header)
    print("-" * (27 + 21 * len(all_metrics)))

    for site in all_sites:
        n_spikes = 0
        for m in all_metrics:
            if site in m["sites"] and m["sites"][site]["n_actual_spikes"]:
                n_spikes = m["sites"][site]["n_actual_spikes"]
                break
        row = f"{site:<20} {n_spikes:>7}"
        for m in all_metrics:
            if site in m["sites"]:
                sr = m["sites"][site]["spike_recall"]
                row += f" {_fmt(sr):>20}"
            else:
                row += f" {'N/A':>20}"
        print(row)

    # Pooling details
    print("\n" + "=" * 90)
    print("POOLING DETAILS (training sample counts)")
    print("=" * 90)
    for m in all_metrics:
        if "pooling" not in m["label"].lower() and "pool" not in m["label"].lower():
            continue
        # Check if we have per-site data with training_samples
        print(f"\n  {m['label']}:")


def _fmt(val) -> str:
    if val is None:
        return "N/A"
    return f"{val:.4f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="A/B test: cross-site pooling + quantile regression"
    )
    parser.add_argument(
        "--sites", nargs="+", default=None,
        help="Specific sites to test (default: all 10)",
    )
    parser.add_argument(
        "--n-anchors", type=int, default=None,
        help="Max test points per site (default: 20%% of measurements)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke test: 5 points per site",
    )
    parser.add_argument(
        "--quantile-alpha", type=float, default=0.7,
        help="Quantile alpha for experiment C (default: 0.7)",
    )
    parser.add_argument(
        "--output", type=str, default="test_results",
        help="Output directory for results (default: test_results/)",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip baseline run (useful if re-running experiments)",
    )
    args = parser.parse_args()

    if args.quick:
        args.n_anchors = 5

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("DATect A/B Test: Cross-Site Pooling + Quantile Regression")
    print("=" * 60)
    print(f"  Sites:           {args.sites or 'all 10'}")
    print(f"  Max per site:    {args.n_anchors or '20% of measurements'}")
    print(f"  Quantile alpha:  {args.quantile_alpha}")
    print(f"  Output:          {output_dir}/")
    print()

    # Initialize engine
    print("Initializing engine...")
    engine = RawForecastEngine(validate_on_init=True)

    # Sample test points (same for all experiments — fair comparison)
    print("Sampling test points...")
    test_points = sample_test_points(
        engine, sites=args.sites, n_per_site=args.n_anchors
    )
    print(f"  {len(test_points)} test points across {test_points['site'].nunique()} sites")
    for site, count in test_points.groupby("site").size().items():
        print(f"    {site}: {count} points")
    print()

    experiments = []

    # --- Experiment A: Baseline ---
    if not args.skip_baseline:
        print("Running Experiment A: Baseline (no pooling, MSE)...")
        df_a = run_experiment(
            engine, test_points,
            label="A: Baseline",
            use_pooled=False, quantile_alpha=None,
        )
        df_a.to_csv(output_dir / "baseline.csv", index=False)
        experiments.append(("A: Baseline", df_a))

    # --- Experiment B: Pooling only ---
    print("\nRunning Experiment B: Cross-site pooling...")
    df_b = run_experiment(
        engine, test_points,
        label="B: Pooling",
        use_pooled=True, quantile_alpha=None,
    )
    df_b.to_csv(output_dir / "pooling.csv", index=False)
    experiments.append(("B: Pooling", df_b))

    # --- Experiment C: Pooling + Quantile ---
    print(f"\nRunning Experiment C: Pooling + Quantile (alpha={args.quantile_alpha})...")
    df_c = run_experiment(
        engine, test_points,
        label=f"C: Pool+Q{int(args.quantile_alpha*100)}",
        use_pooled=True, quantile_alpha=args.quantile_alpha,
    )
    df_c.to_csv(output_dir / "pooling_quantile.csv", index=False)
    experiments.append((f"C: Pool+Q{int(args.quantile_alpha*100)}", df_c))

    # --- Compute and compare metrics ---
    print("\n\nComputing metrics...")
    all_metrics = []
    for label, df in experiments:
        m = compute_metrics(df, label)
        all_metrics.append(m)

    print_comparison(all_metrics)

    # Save metrics as JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to {metrics_path}")

    # Save training sample comparison for pooled sites
    pooled_sites = ["Cannon Beach", "Newport", "Coos Bay", "Gold Beach"]
    print("\n" + "=" * 60)
    print("TRAINING SAMPLE COUNTS (pooled sites)")
    print("=" * 60)
    for label, df in experiments:
        if df.empty:
            continue
        pooled_df = df[df["site"].isin(pooled_sites)]
        if pooled_df.empty:
            continue
        print(f"\n  {label}:")
        for site, sdf in pooled_df.groupby("site"):
            mean_n = sdf["training_samples"].mean()
            was_pooled = sdf["pooled_training"].any() if "pooled_training" in sdf else False
            print(f"    {site}: avg {mean_n:.0f} training samples (pooled={was_pooled})")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

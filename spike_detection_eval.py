"""
Phase 2: Spike Detection Evaluation

Evaluates each model type's ability to detect DA spike events (crossing 20 µg/g
from below). Tests the hypothesis that ML significantly outperforms naive
persistence on spike transitions, even though overall R² is similar.

Usage (Hyak recommended, or locally with cached results):
    python3 spike_detection_eval.py [--force-rerun]
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(__file__))
import config
from forecasting.raw_data_forecaster import load_raw_da_measurements

SPIKE_THRESHOLD = config.SPIKE_THRESHOLD  # 20.0
OUTPUT_DIR = os.path.join("eval_results", "spike_metrics")
RETRO_CACHE_DIR = os.path.join("eval_results", "retro")
WA_SITES = {"Copalis", "Kalaloch", "Quinault", "Twin Harbors", "Long Beach"}

# Model prediction columns available after retrospective runs
MODEL_CONFIGS = {
    "ensemble": {"source_run": "xgb", "source_col": "ensemble_prediction"},
    "xgboost": {"source_run": "xgb", "source_col": "predicted_da"},
    "rf": {"source_run": "xgb", "source_col": "predicted_da_rf"},
    "naive": {"source_run": "xgb", "source_col": "naive_prediction"},
    "linear": {"source_run": "linear", "source_col": "predicted_da_linear"},
}


def load_or_run_retrospective(
    model_type: str, force_rerun: bool = False
) -> pd.DataFrame:
    """Load cached retrospective results or run fresh evaluation."""
    os.makedirs(RETRO_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(RETRO_CACHE_DIR, f"retro_regression_{model_type}.parquet")

    if os.path.exists(cache_path) and not force_rerun:
        print(f"  Loading cached {model_type} results from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"  Running retrospective evaluation for model_type='{model_type}'...")
    print(f"  (This may take 15-45 minutes on Hyak)")

    from forecasting.raw_forecast_engine import RawForecastEngine

    engine = RawForecastEngine()
    results_df = engine.run_retrospective_evaluation(
        task="regression",
        model_type=model_type,
    )

    if results_df is not None and not results_df.empty:
        results_df.to_parquet(cache_path, index=False)
        print(f"  Saved {len(results_df)} results to {cache_path}")

    return results_df


def identify_spike_transitions(
    results_df: pd.DataFrame,
    raw_da: pd.DataFrame,
) -> pd.DataFrame:
    """Flag which test-set predictions correspond to spike transition events.

    For each test row, find the previous raw observation at the same site
    and check if DA crossed from below to above threshold.
    """
    results_df = results_df.copy()
    raw_da = raw_da.sort_values(["site", "date"])

    prev_da_values = []
    prev_dates = []

    for _, row in results_df.iterrows():
        site = row["site"]
        test_date = pd.Timestamp(row["date"])

        # Find the most recent raw observation BEFORE this test date
        site_raw = raw_da[
            (raw_da["site"] == site) & (raw_da["date"] < test_date)
        ]
        if len(site_raw) > 0:
            last_obs = site_raw.iloc[-1]
            prev_da_values.append(last_obs["da_raw"])
            prev_dates.append(last_obs["date"])
        else:
            prev_da_values.append(np.nan)
            prev_dates.append(pd.NaT)

    results_df["prev_da"] = prev_da_values
    results_df["prev_date"] = prev_dates

    # Flag transition types
    actual = results_df["actual_da"]
    prev = results_df["prev_da"]

    results_df["is_spike_transition"] = (prev < SPIKE_THRESHOLD) & (
        actual >= SPIKE_THRESHOLD
    )
    results_df["is_actual_spike"] = actual >= SPIKE_THRESHOLD
    results_df["is_non_spike"] = actual < SPIKE_THRESHOLD
    results_df["gap_days"] = (
        pd.to_datetime(results_df["date"]) - pd.to_datetime(results_df["prev_date"])
    ).dt.days

    return results_df


def compute_spike_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    threshold: float = SPIKE_THRESHOLD,
) -> dict:
    """Compute binary spike detection metrics."""
    actual_binary = (actual >= threshold).astype(int)
    predicted_binary = (predicted >= threshold).astype(int)

    tp = int(((actual_binary == 1) & (predicted_binary == 1)).sum())
    fn = int(((actual_binary == 1) & (predicted_binary == 0)).sum())
    fp = int(((actual_binary == 0) & (predicted_binary == 1)).sum())
    tn = int(((actual_binary == 0) & (predicted_binary == 0)).sum())

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # F2 weights recall 2x higher
    f2_beta = 2.0
    f2 = (
        (1 + f2_beta**2) * precision * recall / (f2_beta**2 * precision + recall)
        if (f2_beta**2 * precision + recall) > 0
        else 0.0
    )

    # MAE stratified
    spike_mask = actual >= threshold
    mae_spikes = float(np.abs(actual[spike_mask] - predicted[spike_mask]).mean()) if spike_mask.sum() > 0 else np.nan
    mae_non_spikes = float(np.abs(actual[~spike_mask] - predicted[~spike_mask]).mean()) if (~spike_mask).sum() > 0 else np.nan

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "n_actual_spikes": int(actual_binary.sum()),
        "n_predicted_spikes": int(predicted_binary.sum()),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "f2": f2,
        "fnr": 1.0 - recall,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "mae_on_spikes": mae_spikes,
        "mae_on_non_spikes": mae_non_spikes,
    }


def compute_transition_metrics(
    results_df: pd.DataFrame,
    pred_col: str,
    threshold: float = SPIKE_THRESHOLD,
) -> dict:
    """Compute metrics specifically on spike transition events."""
    transitions = results_df[results_df["is_spike_transition"]].copy()
    if len(transitions) == 0:
        return {"transition_n": 0, "transition_recall": np.nan}

    actual = transitions["actual_da"].values
    predicted = transitions[pred_col].values

    predicted_spike = predicted >= threshold
    actual_spike = actual >= threshold  # should all be True by definition

    transition_recall = predicted_spike.sum() / len(transitions) if len(transitions) > 0 else 0.0

    return {
        "transition_n": int(len(transitions)),
        "transition_recall": float(transition_recall),
        "transition_mean_predicted": float(predicted.mean()),
        "transition_mean_actual": float(actual.mean()),
        "transition_mae": float(np.abs(actual - predicted).mean()),
    }


def run_threshold_sensitivity(
    results_df: pd.DataFrame,
    pred_col: str,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Sweep prediction decision thresholds while keeping actual spike def at 20."""
    if thresholds is None:
        thresholds = [8, 10, 12, 14, 15, 16, 18, 20, 22, 25, 30]

    actual = results_df["actual_da"].values
    predicted = results_df[pred_col].values

    rows = []
    for thresh in thresholds:
        metrics = compute_spike_metrics(actual, predicted, threshold=SPIKE_THRESHOLD)
        # But override the prediction threshold
        pred_binary = (predicted >= thresh).astype(int)
        actual_binary = (actual >= SPIKE_THRESHOLD).astype(int)

        tp = int(((actual_binary == 1) & (pred_binary == 1)).sum())
        fn = int(((actual_binary == 1) & (pred_binary == 0)).sum())
        fp = int(((actual_binary == 0) & (pred_binary == 1)).sum())
        tn = int(((actual_binary == 0) & (pred_binary == 0)).sum())

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f2_beta = 2.0
        f2 = (
            (1 + f2_beta**2) * precision * recall / (f2_beta**2 * precision + recall)
            if (f2_beta**2 * precision + recall) > 0
            else 0.0
        )

        rows.append({
            "pred_threshold": thresh,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "f2": f2,
            "tp": tp,
            "fn": fn,
            "fp": fp,
        })

    return pd.DataFrame(rows)


def plot_model_comparison(metrics_df: pd.DataFrame, output_path: str):
    """Bar chart comparing models on spike detection metrics."""
    models = metrics_df["model"].values
    metric_names = ["recall", "precision", "f1", "f2", "transition_recall"]
    metric_labels = ["Spike Recall", "Spike Precision", "F1", "F2 (recall-weighted)", "Transition Recall"]

    x = np.arange(len(metric_names))
    width = 0.15
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    for i, (_, row) in enumerate(metrics_df.iterrows()):
        vals = [row.get(m, 0) for m in metric_names]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=row["model"], color=colors[i % len(colors)])
        # Add value labels
        for bar, val in zip(bars, vals):
            if pd.notna(val) and val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Spike Detection Performance by Model\n(actual spike = DA >= 20 µg/g)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved model comparison plot to {output_path}")


def plot_threshold_sweep(sweep_df: pd.DataFrame, output_path: str):
    """Plot threshold sensitivity analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sweep_df["pred_threshold"], sweep_df["recall"], "o-", label="Recall", linewidth=2)
    ax.plot(sweep_df["pred_threshold"], sweep_df["precision"], "s-", label="Precision", linewidth=2)
    ax.plot(sweep_df["pred_threshold"], sweep_df["f1"], "^-", label="F1", linewidth=2)
    ax.plot(sweep_df["pred_threshold"], sweep_df["f2"], "D-", label="F2 (recall-weighted)", linewidth=2, color="red")

    # Mark the default threshold
    ax.axvline(x=20, color="gray", linestyle="--", alpha=0.5, label="Default threshold (20)")

    # Mark optimal F2
    best_f2_idx = sweep_df["f2"].idxmax()
    best_thresh = sweep_df.loc[best_f2_idx, "pred_threshold"]
    best_f2_val = sweep_df.loc[best_f2_idx, "f2"]
    ax.annotate(
        f"Best F2={best_f2_val:.3f}\nat threshold={best_thresh}",
        xy=(best_thresh, best_f2_val),
        xytext=(best_thresh + 3, best_f2_val - 0.1),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9,
        color="red",
    )

    ax.set_xlabel("Prediction Decision Threshold (µg/g)")
    ax.set_ylabel("Score")
    ax.set_title("Ensemble Spike Detection: Threshold Sensitivity\n(Actual spike always defined as DA >= 20 µg/g)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved threshold sweep plot to {output_path}")


def print_summary(all_metrics: pd.DataFrame, per_site_metrics: pd.DataFrame):
    """Print human-readable summary."""
    print("\n" + "=" * 80)
    print("PHASE 2: SPIKE DETECTION EVALUATION — SUMMARY")
    print("=" * 80)

    # --- Confusion matrix breakdown for each model ---
    print("\n--- Confusion Matrix Breakdown (DA >= 20 threshold) ---")
    print(f"  {'Model':<22s}  {'TP':>4s}  {'FP':>4s}  {'FN':>4s}  {'TN':>4s}  "
          f"{'Alerts':>6s}  {'Actual':>6s}  {'Recall':>7s}  {'Precision':>9s}  "
          f"{'Trans.':>6s}")
    print("  " + "-" * 95)
    for _, row in all_metrics.iterrows():
        tp = int(row.get("tp", 0))
        fp = int(row.get("fp", 0))
        fn = int(row.get("fn", 0))
        tn = int(row.get("tn", 0))
        n_alerts = tp + fp
        n_actual = tp + fn
        recall = row.get("recall", 0)
        precision = row.get("precision", 0)
        trans = row.get("transition_recall", float("nan"))
        trans_str = f"{trans:.3f}" if not np.isnan(trans) else "  n/a"
        print(f"  {row['model']:<22s}  {tp:4d}  {fp:4d}  {fn:4d}  {tn:4d}  "
              f"{n_alerts:6d}  {n_actual:6d}  {recall:7.3f}  {precision:9.3f}  "
              f"{trans_str:>6s}")
    print()
    print("  TP = correctly alerted spikes     FP = false alarms (alert but no spike)")
    print("  FN = missed spikes (no alert)     TN = correctly quiet")
    print("  Alerts = total alerts raised       Actual = real spike events")
    print("  Trans. = transition recall (caught spikes where prev obs was < 20)")

    # --- Overall summary table ---
    print("\n--- Overall Spike Detection Metrics ---")
    cols = ["model", "n_actual_spikes", "recall", "precision", "f1", "f2", "fnr", "transition_recall"]
    display_cols = [c for c in cols if c in all_metrics.columns]
    print(all_metrics[display_cols].to_string(index=False, float_format="%.3f"))

    # --- Spike classifier detail (if present) ---
    spike_rows = all_metrics[all_metrics["model"].str.startswith("spike_classifier")]
    if len(spike_rows) > 0:
        sr = spike_rows.iloc[0]
        tp = int(sr.get("tp", 0))
        fp = int(sr.get("fp", 0))
        fn = int(sr.get("fn", 0))
        n_alerts = tp + fp
        trans_n = int(sr.get("transition_n", 0))
        trans_recall = sr.get("transition_recall", 0)
        trans_caught = int(round(trans_recall * trans_n)) if trans_n > 0 else 0
        print(f"\n--- Spike Classifier Breakdown ---")
        print(f"  Raised {n_alerts} alerts total:")
        print(f"    {tp} were real spikes (true positives)")
        print(f"    {fp} were false alarms (DA stayed below 20)")
        print(f"  Missed {fn} actual spikes (false negatives)")
        print(f"  Of {trans_n} NEW spike transitions: caught {trans_caught} ({trans_recall:.1%})")
        print(f"  Of {int(sr.get('n_actual_spikes', 0))} total spike events: caught {tp} ({sr.get('recall', 0):.1%})")

    # --- Hypothesis test ---
    print("\n--- Hypothesis Test: Does ML outperform naive on spike detection? ---")
    naive_row = all_metrics[all_metrics["model"] == "naive"].iloc[0] if "naive" in all_metrics["model"].values else None
    ensemble_row = all_metrics[all_metrics["model"] == "ensemble"].iloc[0] if "ensemble" in all_metrics["model"].values else None

    if naive_row is not None and ensemble_row is not None:
        print(f"\n  Naive spike recall:    {naive_row['recall']:.3f}")
        print(f"  Ensemble spike recall: {ensemble_row['recall']:.3f}")
        print(f"\n  Naive transition recall:    {naive_row['transition_recall']:.3f}")
        print(f"  Ensemble transition recall: {ensemble_row['transition_recall']:.3f}")

        # The real comparison: spike classifier vs naive on transitions
        if len(spike_rows) > 0:
            sc_trans = spike_rows.iloc[0].get("transition_recall", 0)
            naive_trans = naive_row["transition_recall"]
            improvement = sc_trans / naive_trans if naive_trans > 0 else float("inf")
            print(f"\n  Spike classifier transition recall: {sc_trans:.3f}")
            print(f"  vs naive transition recall:         {naive_trans:.3f}")
            print(f"  Improvement factor:                 {improvement:.1f}x")

            if sc_trans > naive_trans + 0.10:
                print("\n  RESULT: Spike classifier significantly outperforms naive on transitions.")
                print("  The dedicated classifier provides advance warning for the events that")
                print("  matter most — new spike onsets that persistence-based methods miss.")
            elif sc_trans > naive_trans:
                print("\n  RESULT: Spike classifier shows modest improvement over naive on transitions.")
            else:
                print("\n  RESULT: Spike classifier does not outperform naive on transitions.")

    # WA sites breakdown
    wa_metrics = per_site_metrics[per_site_metrics["site"].isin(WA_SITES)]
    if len(wa_metrics) > 0:
        print("\n--- Washington Sites (Primary Focus) ---")
        wa_display = wa_metrics[["site", "model", "n_actual_spikes", "recall", "f2", "transition_recall"]]
        print(wa_display.to_string(index=False, float_format="%.3f"))

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Spike Detection Evaluation")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-run of retrospective evaluation")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed (e.g., 123 for independent test set)")
    parser.add_argument("--sample-fraction", type=float, default=None,
                        help="Override config.TEST_SAMPLE_FRACTION (e.g. 0.40 for 40%% independent test set)")
    args = parser.parse_args()

    # Override seed before any engine initialization
    if args.seed is not None:
        config.RANDOM_SEED = args.seed
        print(f"Using seed={args.seed} (independent test set)")

    if args.sample_fraction is not None:
        config.TEST_SAMPLE_FRACTION = args.sample_fraction
        print(f"Using sample_fraction={args.sample_fraction} (overrides config default of {config.TEST_SAMPLE_FRACTION})")
    print(f"Test sample fraction: {config.TEST_SAMPLE_FRACTION}")

    seed_suffix = f"_seed{config.RANDOM_SEED}" if config.RANDOM_SEED != 42 else ""

    # Use seed-specific output and cache directories
    global OUTPUT_DIR, RETRO_CACHE_DIR
    OUTPUT_DIR = os.path.join("eval_results", f"spike_metrics{seed_suffix}")
    RETRO_CACHE_DIR = os.path.join("eval_results", f"retro{seed_suffix}")

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load raw DA for spike transition identification
    print("Loading raw DA measurements...")
    raw_da = load_raw_da_measurements()
    raw_da["date"] = pd.to_datetime(raw_da["date"])

    # Run/load retrospective evaluations
    # Run 1: model_type="xgb" saves four columns: XGB, RF, naive, and ensemble.
    #         naive_prediction is computed as a standalone external baseline (w_naive=0.0
    #         everywhere, so it does NOT contribute to ensemble_prediction).
    # Run 2: model_type="linear" gives linear predictions
    print("\nLoading retrospective results...")
    print("  Run 1: XGB run (saves XGB, RF, naive, and ensemble columns; naive is external baseline, not blended)...")
    xgb_results = load_or_run_retrospective("xgb", force_rerun=args.force_rerun)
    print("  Run 2: Linear...")
    linear_results = load_or_run_retrospective("linear", force_rerun=args.force_rerun)

    if xgb_results is None or xgb_results.empty:
        print("ERROR: No retrospective results available. Run on Hyak first.")
        sys.exit(1)

    # Ensure date columns are datetime
    for df in [xgb_results, linear_results] if linear_results is not None else [xgb_results]:
        df["date"] = pd.to_datetime(df["date"])

    # Merge linear predictions into main results
    results = xgb_results.copy()
    if linear_results is not None and not linear_results.empty:
        # linear run has predicted_da_linear; but predicted_da was overwritten to linear
        # We need the predicted_da_linear column from the linear run
        if "predicted_da_linear" in linear_results.columns:
            linear_preds = linear_results[["date", "site", "predicted_da_linear"]].copy()
        else:
            # predicted_da in linear run IS the linear prediction
            linear_preds = linear_results[["date", "site", "predicted_da"]].rename(
                columns={"predicted_da": "predicted_da_linear"}
            )
        results = results.merge(linear_preds, on=["date", "site"], how="left", suffixes=("", "_lin"))

    # Identify spike transitions
    print("\nIdentifying spike transitions in test set...")
    results = identify_spike_transitions(results, raw_da)

    n_transitions = results["is_spike_transition"].sum()
    n_spikes = results["is_actual_spike"].sum()
    print(f"  Test set size: {len(results)}")
    print(f"  Actual spike events (DA >= 20): {n_spikes}")
    print(f"  Spike transitions (crossed from below): {n_transitions}")

    # Compute metrics for each model
    print("\nComputing spike detection metrics...")
    all_metrics = []
    per_site_metrics = []

    for model_name, cfg in MODEL_CONFIGS.items():
        col = cfg["source_col"]
        if col not in results.columns:
            print(f"  Skipping {model_name}: column '{col}' not available")
            continue

        valid = results[results[col].notna()].copy()
        if len(valid) == 0:
            continue

        # Overall metrics
        metrics = compute_spike_metrics(valid["actual_da"].values, valid[col].values)
        trans_metrics = compute_transition_metrics(valid, col)
        metrics.update(trans_metrics)
        metrics["model"] = model_name
        metrics["n_test_points"] = len(valid)
        all_metrics.append(metrics)

        # Per-site metrics
        for site in sorted(valid["site"].unique()):
            site_data = valid[valid["site"] == site]
            if len(site_data) < 5:
                continue
            site_metrics = compute_spike_metrics(
                site_data["actual_da"].values, site_data[col].values
            )
            site_trans = compute_transition_metrics(site_data, col)
            site_metrics.update(site_trans)
            site_metrics["model"] = model_name
            site_metrics["site"] = site
            per_site_metrics.append(site_metrics)

    # Evaluate integrated spike binary classifier (if present in results)
    if "spike_probability" in results.columns and results["spike_probability"].notna().sum() > 0:
        print("\n  Evaluating integrated spike classifier...")
        prob_threshold = getattr(config, "SPIKE_ALERT_PROB_THRESHOLD", 0.10)
        valid_spike = results[results["spike_probability"].notna()].copy()
        # Convert probability to binary prediction
        spike_pred_binary = (valid_spike["spike_probability"] >= prob_threshold).astype(int)
        actual_binary = (valid_spike["actual_da"] >= SPIKE_THRESHOLD).astype(int)

        tp = int(((actual_binary == 1) & (spike_pred_binary == 1)).sum())
        fn = int(((actual_binary == 1) & (spike_pred_binary == 0)).sum())
        fp = int(((actual_binary == 0) & (spike_pred_binary == 1)).sum())
        tn = int(((actual_binary == 0) & (spike_pred_binary == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f2_beta = 2.0
        f2 = (
            (1 + f2_beta**2) * precision * recall / (f2_beta**2 * precision + recall)
            if (f2_beta**2 * precision + recall) > 0 else 0.0
        )

        # Transition recall for spike classifier
        transitions = valid_spike[valid_spike["is_spike_transition"]].copy()
        if len(transitions) > 0:
            trans_pred = (transitions["spike_probability"] >= prob_threshold).astype(int)
            trans_recall = float(trans_pred.sum()) / len(transitions)
        else:
            trans_recall = np.nan

        spike_cls_metrics = {
            "model": f"spike_classifier_p{int(prob_threshold*100):02d}",
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "n_actual_spikes": int(actual_binary.sum()),
            "n_predicted_spikes": int(spike_pred_binary.sum()),
            "recall": recall, "precision": precision, "f1": f1, "f2": f2,
            "fnr": 1.0 - recall,
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            "mae_on_spikes": np.nan, "mae_on_non_spikes": np.nan,
            "transition_n": len(transitions),
            "transition_recall": trans_recall,
            "transition_mean_predicted": np.nan,
            "transition_mean_actual": float(transitions["actual_da"].mean()) if len(transitions) > 0 else np.nan,
            "transition_mae": np.nan,
            "n_test_points": len(valid_spike),
        }
        all_metrics.append(spike_cls_metrics)
        print(f"    Spike classifier (prob>={prob_threshold:.2f}): recall={recall:.3f}, "
              f"transition_recall={trans_recall:.3f}, F2={f2:.3f}")

        # Per-site metrics for spike classifier
        sc_model_name = f"spike_classifier_p{int(prob_threshold*100):02d}"
        for site in sorted(valid_spike["site"].unique()):
            site_data = valid_spike[valid_spike["site"] == site]
            if len(site_data) < 3:
                continue
            s_pred = (site_data["spike_probability"] >= prob_threshold).astype(int)
            s_actual = (site_data["actual_da"] >= SPIKE_THRESHOLD).astype(int)
            s_tp = int(((s_actual == 1) & (s_pred == 1)).sum())
            s_fn = int(((s_actual == 1) & (s_pred == 0)).sum())
            s_fp = int(((s_actual == 0) & (s_pred == 1)).sum())
            s_tn = int(((s_actual == 0) & (s_pred == 0)).sum())
            s_recall = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
            s_prec = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
            s_f2 = (5 * s_prec * s_recall / (4 * s_prec + s_recall)
                    if (4 * s_prec + s_recall) > 0 else 0.0)
            # Transition recall
            s_trans = site_data[site_data["is_spike_transition"]]
            if len(s_trans) > 0:
                s_trans_recall = float((s_trans["spike_probability"] >= prob_threshold).sum()) / len(s_trans)
            else:
                s_trans_recall = np.nan
            per_site_metrics.append({
                "model": sc_model_name, "site": site,
                "tp": s_tp, "fp": s_fp, "fn": s_fn, "tn": s_tn,
                "n_actual_spikes": int(s_actual.sum()),
                "recall": s_recall, "precision": s_prec, "f2": s_f2,
                "transition_recall": s_trans_recall,
                "transition_n": len(s_trans),
            })

    all_metrics_df = pd.DataFrame(all_metrics)
    per_site_df = pd.DataFrame(per_site_metrics)

    # Save metrics
    all_metrics_df.to_csv(os.path.join(OUTPUT_DIR, "spike_metrics_by_model.csv"), index=False)
    per_site_df.to_csv(os.path.join(OUTPUT_DIR, "per_site_metrics.csv"), index=False)

    # Threshold sensitivity for ensemble
    if "ensemble_prediction" in results.columns:
        print("\nRunning threshold sensitivity analysis for ensemble...")
        valid_ens = results[results["ensemble_prediction"].notna()]
        sweep_df = run_threshold_sensitivity(valid_ens, "ensemble_prediction")
        sweep_df.to_csv(os.path.join(OUTPUT_DIR, "threshold_sweep.csv"), index=False)
        plot_threshold_sweep(sweep_df, os.path.join(OUTPUT_DIR, "threshold_sweep.png"))

    # Visualizations
    print("\nGenerating comparison plots...")
    plot_model_comparison(all_metrics_df, os.path.join(OUTPUT_DIR, "model_comparison.png"))

    # Summary
    print_summary(all_metrics_df, per_site_df)


if __name__ == "__main__":
    main()

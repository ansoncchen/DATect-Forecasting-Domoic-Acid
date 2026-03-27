"""
Part A: Spike Transition Analysis — Prototype

Two analyses:
1. Threshold sweep with TRANSITION-SPECIFIC metrics (the Phase 2 sweep only
   measured overall spike recall, not transition recall)
2. Prototype binary spike classifier trained on features from the feature frame

Usage (Hyak, uses cached retrospective results):
    python3 spike_transition_analysis.py
"""

import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(__file__))
import config
from forecasting.raw_data_forecaster import (
    load_raw_da_measurements,
    build_raw_feature_frame,
    RawForecastConfig,
)
from forecasting.raw_data_processor import RawDataProcessor

SPIKE_THRESHOLD = config.SPIKE_THRESHOLD  # 20.0
OUTPUT_DIR = os.path.join("eval_results", "spike_transition")
RETRO_CACHE = os.path.join("eval_results", "retro", "retro_regression_xgb.parquet")

# Features to drop from classifier training (same as regression pipeline)
DROP_COLS = {"date", "site", "da_raw", "da", "_is_interpolated"}
ZERO_IMPORTANCE = set(getattr(config, "ZERO_IMPORTANCE_FEATURES", []))


# =============================================================================
# Part 1: Transition-Specific Threshold Sweep
# =============================================================================


def identify_transitions(results_df: pd.DataFrame, raw_da: pd.DataFrame) -> pd.DataFrame:
    """Flag spike transition events in results. Reuses logic from spike_detection_eval."""
    results_df = results_df.copy()
    raw_da = raw_da.sort_values(["site", "date"])

    prev_da_values = []
    for _, row in results_df.iterrows():
        site = row["site"]
        test_date = pd.Timestamp(row["date"])
        site_raw = raw_da[(raw_da["site"] == site) & (raw_da["date"] < test_date)]
        if len(site_raw) > 0:
            prev_da_values.append(site_raw.iloc[-1]["da_raw"])
        else:
            prev_da_values.append(np.nan)

    results_df["prev_da"] = prev_da_values
    results_df["is_spike_transition"] = (
        (results_df["prev_da"] < SPIKE_THRESHOLD)
        & (results_df["actual_da"] >= SPIKE_THRESHOLD)
    )
    results_df["is_actual_spike"] = results_df["actual_da"] >= SPIKE_THRESHOLD
    return results_df


def transition_threshold_sweep(
    results_df: pd.DataFrame,
    pred_col: str,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Sweep prediction thresholds measuring TRANSITION-specific metrics."""
    if thresholds is None:
        thresholds = [6, 8, 10, 12, 14, 15, 16, 18, 20, 22, 25, 30]

    actual = results_df["actual_da"].values
    predicted = results_df[pred_col].values
    is_transition = results_df["is_spike_transition"].values
    is_spike = results_df["is_actual_spike"].values

    rows = []
    for thresh in thresholds:
        pred_spike = predicted >= thresh
        actual_spike = actual >= SPIKE_THRESHOLD

        # Overall metrics
        tp = int((actual_spike & pred_spike).sum())
        fn = int((actual_spike & ~pred_spike).sum())
        fp = int((~actual_spike & pred_spike).sum())
        tn = int((~actual_spike & ~pred_spike).sum())

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        beta = 2.0
        f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else 0.0

        # Transition-specific
        trans_mask = is_transition
        n_trans = int(trans_mask.sum())
        if n_trans > 0:
            trans_caught = int((trans_mask & pred_spike).sum())
            trans_recall = trans_caught / n_trans
        else:
            trans_caught = 0
            trans_recall = np.nan

        # Transition F2: treating transitions as positives
        trans_tp = trans_caught
        trans_fn = n_trans - trans_caught
        # False positives: predicted spike but actual < 20 (same as overall fp)
        trans_precision = trans_tp / (trans_tp + fp) if (trans_tp + fp) > 0 else 0.0
        trans_f1 = (
            2 * trans_precision * trans_recall / (trans_precision + trans_recall)
            if pd.notna(trans_recall) and (trans_precision + trans_recall) > 0
            else 0.0
        )
        trans_f2 = (
            (1 + beta**2) * trans_precision * trans_recall / (beta**2 * trans_precision + trans_recall)
            if pd.notna(trans_recall) and (beta**2 * trans_precision + trans_recall) > 0
            else 0.0
        )

        rows.append({
            "pred_threshold": thresh,
            "overall_recall": recall,
            "overall_precision": precision,
            "overall_f1": f1,
            "overall_f2": f2,
            "transition_recall": trans_recall,
            "transition_precision": trans_precision,
            "transition_f1": trans_f1,
            "transition_f2": trans_f2,
            "n_transitions": n_trans,
            "transitions_caught": trans_caught,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Part 2: Prototype Binary Spike Classifier
# =============================================================================


def build_feature_frame_for_classifier():
    """Build the full feature frame (same as forecasting engine uses)."""
    print("  Building feature frame...")
    env_data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
    env_data["date"] = pd.to_datetime(env_data["date"])

    raw_da = load_raw_da_measurements()
    raw_da["date"] = pd.to_datetime(raw_da["date"])

    # Aggregate raw DA to weekly (same as engine does)
    raw_weekly = raw_da.groupby(["site", "date"]).agg({"da_raw": "mean"}).reset_index()
    raw_weekly["date"] = pd.to_datetime(raw_weekly["date"])

    # Snap to weekly grid
    grid_dates = sorted(env_data["date"].unique())

    def snap_to_grid(d):
        diffs = [abs((gd - d).days) for gd in grid_dates]
        min_idx = np.argmin(diffs)
        if diffs[min_idx] <= 4:
            return grid_dates[min_idx]
        return pd.NaT

    raw_weekly["date"] = raw_weekly["date"].apply(snap_to_grid)
    raw_weekly = raw_weekly.dropna(subset=["date"])
    # Dedup: keep mean if multiple obs snap to same grid date
    raw_weekly = raw_weekly.groupby(["site", "date"]).agg({"da_raw": "mean"}).reset_index()

    feature_frame = build_raw_feature_frame(env_data, raw_weekly)

    # Add new spike-proximity features
    feature_frame["distance_to_threshold"] = (
        SPIKE_THRESHOLD - feature_frame["last_observed_da_raw"]
    )

    # Environmental risk composite (z-scored per site)
    for feat in ["modis-sst", "pdo", "discharge"]:
        if feat in feature_frame.columns:
            mean = feature_frame.groupby("site")[feat].transform("mean")
            std = feature_frame.groupby("site")[feat].transform("std")
            std = std.replace(0, 1)
            feature_frame[f"{feat}_z"] = (feature_frame[feat] - mean) / std

    risk_cols = []
    if "modis-sst_z" in feature_frame.columns:
        risk_cols.append("modis-sst_z")
    if "pdo_z" in feature_frame.columns:
        risk_cols.append("pdo_z")

    if risk_cols:
        feature_frame["env_risk_composite"] = feature_frame[risk_cols].sum(axis=1)
        if "discharge_z" in feature_frame.columns:
            feature_frame["env_risk_composite"] -= feature_frame["discharge_z"]

    # DA acceleration
    if (
        "da_raw_prev_obs_diff_1_2" in feature_frame.columns
        and "da_raw_prev_obs_2" in feature_frame.columns
        and "da_raw_prev_obs_3" in feature_frame.columns
    ):
        diff_2_3 = feature_frame["da_raw_prev_obs_2"] - feature_frame["da_raw_prev_obs_3"]
        feature_frame["da_acceleration"] = (
            feature_frame["da_raw_prev_obs_diff_1_2"] - diff_2_3
        )

    print(f"  Feature frame: {feature_frame.shape[0]} rows, {feature_frame.shape[1]} columns")
    return feature_frame


def train_spike_classifier(
    feature_frame: pd.DataFrame,
    test_points: pd.DataFrame,
) -> dict:
    """Train a binary spike classifier and evaluate on test points.

    Returns dict with predictions and metrics.
    """
    ff = feature_frame.copy()
    ff["date"] = pd.to_datetime(ff["date"])
    test_points = test_points.copy()
    test_points["date"] = pd.to_datetime(test_points["date"])

    # Binary target: is DA >= 20?
    ff["spike_target"] = (ff["da_raw"] >= SPIKE_THRESHOLD).astype(int)

    # Identify test rows by matching date + site
    ff["_is_test"] = False
    for _, tp in test_points.iterrows():
        mask = (ff["site"] == tp["site"]) & (ff["date"] == tp["date"])
        ff.loc[mask, "_is_test"] = True

    # For test rows where exact date didn't match, try closest date
    matched = ff[ff["_is_test"]].shape[0]
    print(f"  Matched {matched}/{len(test_points)} test points in feature frame")

    # If match rate is low, try snapping test dates to nearest grid date
    if matched < len(test_points) * 0.5:
        print("  Low match rate, trying date snapping...")
        grid_dates = sorted(ff["date"].unique())
        for _, tp in test_points.iterrows():
            tp_date = pd.Timestamp(tp["date"])
            diffs = [abs((gd - tp_date).days) for gd in grid_dates]
            min_idx = np.argmin(diffs)
            if diffs[min_idx] <= 7:
                mask = (ff["site"] == tp["site"]) & (ff["date"] == grid_dates[min_idx])
                ff.loc[mask, "_is_test"] = True
        matched = ff[ff["_is_test"]].shape[0]
        print(f"  After snapping: matched {matched}/{len(test_points)}")

    # Fix data leakage: for test rows, persistence features were computed using
    # the current row's da_raw (e.g. last_observed_da_raw = the spike value itself).
    # In the real engine, test row persistence features are recomputed using only
    # training data (dates <= anchor_date). Simulate this by rebuilding persistence
    # features on a version of da_raw where test-row values are masked out.
    ff_leak_free = ff.copy()
    ff_leak_free.loc[ff_leak_free["_is_test"], "da_raw"] = np.nan

    # Recompute persistence features without test-row DA values
    ff_leak_free = ff_leak_free.sort_values(["site", "date"])
    ff_leak_free["last_observed_da_raw"] = ff_leak_free.groupby("site")["da_raw"].ffill()
    ff_leak_free["distance_to_threshold"] = SPIKE_THRESHOLD - ff_leak_free["last_observed_da_raw"]

    last_spike_date = ff_leak_free["date"].where(ff_leak_free["da_raw"] > SPIKE_THRESHOLD)
    last_spike_date = last_spike_date.groupby(ff_leak_free["site"]).ffill()
    ff_leak_free["weeks_since_last_spike"] = (
        (ff_leak_free["date"] - last_spike_date).dt.days / 7.0
    ).fillna(999.0)

    # Also recompute env_risk_composite with new discharge_z (unchanged but re-apply)
    if "env_risk_composite" in ff_leak_free.columns:
        risk_cols = []
        if "modis-sst_z" in ff_leak_free.columns:
            risk_cols.append("modis-sst_z")
        if "pdo_z" in ff_leak_free.columns:
            risk_cols.append("pdo_z")
        if risk_cols:
            ff_leak_free["env_risk_composite"] = ff_leak_free[risk_cols].sum(axis=1)
            if "discharge_z" in ff_leak_free.columns:
                ff_leak_free["env_risk_composite"] -= ff_leak_free["discharge_z"]

    # Restore spike_target (it uses original da_raw, which is correct)
    ff_leak_free["spike_target"] = ff["spike_target"]
    ff_leak_free["_is_test"] = ff["_is_test"]

    # Split
    train_ff = ff_leak_free[~ff_leak_free["_is_test"] & ff["da_raw"].notna()].copy()
    test_ff = ff_leak_free[ff_leak_free["_is_test"] & ff["da_raw"].notna()].copy()

    # Drop non-feature columns (including da_raw itself — it's the target, not a feature)
    drop = DROP_COLS | ZERO_IMPORTANCE | {"spike_target", "_is_test"}
    drop |= {"modis-sst_z", "pdo_z", "discharge_z"}
    # Drop persistence features derived from the CURRENT row's da_raw:
    # last_observed_da_raw = ffill(da_raw), so for training rows it equals da_raw.
    # weeks_since_last_spike and distance_to_threshold depend on it directly.
    # da_raw_prev_obs_1 correctly represents the previous observation and is kept.
    drop |= {"last_observed_da_raw", "weeks_since_last_spike", "distance_to_threshold"}
    feature_cols = [c for c in train_ff.columns if c not in drop and not c.startswith("_")]
    feature_cols = [c for c in feature_cols if train_ff[c].dtype in ("float64", "float32", "int64", "int32")]

    X_train = train_ff[feature_cols].copy()
    y_train = train_ff["spike_target"].values
    X_test = test_ff[feature_cols].copy()
    y_test = test_ff["spike_target"].values

    # Handle NaNs
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    print(f"  Training: {len(X_train)} samples, {y_train.sum()} positive ({y_train.mean():.1%})")
    print(f"  Test: {len(X_test)} samples, {y_test.sum()} positive ({y_test.mean():.1%})")

    # Class weights
    classes = np.unique(y_train)
    if len(classes) < 2:
        print("  ERROR: Only one class in training data")
        return {}

    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = dict(zip(classes, weights))
    sample_weights = np.array([weight_dict[y] for y in y_train])

    # Train XGBoost classifier
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=config.RANDOM_SEED,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

    # Predict probabilities
    proba = clf.predict_proba(X_test)[:, 1]

    # Merge predictions back with test data
    test_ff = test_ff.copy()
    test_ff["spike_probability"] = proba
    test_ff["spike_predicted_20"] = (proba >= 0.5).astype(int)

    # Feature importance
    feat_imp = dict(zip(feature_cols, clf.feature_importances_))
    top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15]

    # --- Transition-focused classifier ---
    # Train only on rows where the PREVIOUS observation was below threshold.
    # Use da_raw_prev_obs_1 (not last_observed_da_raw which equals da_raw for
    # training rows). This correctly includes transition rows (prev < 20, current >= 20).
    prev_col = "da_raw_prev_obs_1"
    if prev_col not in train_ff.columns:
        prev_col = next((c for c in train_ff.columns if "prev_obs_1" in c), None)

    # Compute safe-baseline mask once; reuse for both filtering and index mapping
    if prev_col:
        train_safe_mask = train_ff[prev_col].fillna(0) < SPIKE_THRESHOLD
        test_safe_mask = test_ff[prev_col].fillna(0) < SPIKE_THRESHOLD
    else:
        train_safe_mask = train_ff["last_observed_da_raw"].fillna(0) < SPIKE_THRESHOLD
        test_safe_mask = test_ff["last_observed_da_raw"].fillna(SPIKE_THRESHOLD + 1) < SPIKE_THRESHOLD

    safe_train = train_ff[train_safe_mask].copy()
    safe_test = test_ff[test_safe_mask].copy()
    safe_test_idx = np.where(test_safe_mask.values)[0]  # positions in test_ff

    proba_trans = np.full(len(test_ff), np.nan)
    top_features_trans = []

    if len(safe_train) > 50 and safe_train["spike_target"].sum() >= 5:
        X_train_safe = safe_train[feature_cols].fillna(0)
        y_train_safe = safe_train["spike_target"].values
        X_test_safe = safe_test[feature_cols].fillna(0)

        w_safe = compute_class_weight("balanced", classes=np.unique(y_train_safe), y=y_train_safe)
        sw_safe = np.array([dict(zip(np.unique(y_train_safe), w_safe))[y] for y in y_train_safe])

        clf_trans = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=config.RANDOM_SEED,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        clf_trans.fit(X_train_safe, y_train_safe, sample_weight=sw_safe, verbose=False)

        if len(X_test_safe) > 0:
            proba_safe = clf_trans.predict_proba(X_test_safe)[:, 1]
            # Map back to full test_ff index positions using the same mask
            for i, idx in enumerate(safe_test_idx):
                proba_trans[idx] = proba_safe[i]

        feat_imp_trans = dict(zip(feature_cols, clf_trans.feature_importances_))
        top_features_trans = sorted(feat_imp_trans.items(), key=lambda x: x[1], reverse=True)[:15]
        n_safe_pos = int(safe_train["spike_target"].sum())
        print(f"  Safe-baseline classifier: {len(safe_train)} train ({n_safe_pos} positive), "
              f"{len(safe_test)} test rows where DA was safe before")
    else:
        print(f"  Safe-baseline classifier: insufficient safe-baseline training data "
              f"({len(safe_train)} rows, {safe_train['spike_target'].sum()} positive)")

    return {
        "test_df": test_ff,
        "proba": proba,
        "proba_trans": proba_trans,
        "y_test": y_test,
        "clf": clf,
        "top_features": top_features,
        "top_features_trans": top_features_trans,
        "feature_cols": feature_cols,
    }


def evaluate_classifier_at_thresholds(
    proba: np.ndarray,
    y_test: np.ndarray,
    is_transition: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate classifier at different probability thresholds."""
    if thresholds is None:
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]

    rows = []
    beta = 2.0
    for thresh in thresholds:
        pred = (proba >= thresh).astype(int)
        tp = int(((y_test == 1) & (pred == 1)).sum()) if len(y_test) > 0 else 0
        fn = int(((y_test == 1) & (pred == 0)).sum())
        fp = int(((y_test == 0) & (pred == 1)).sum())
        tn = int(((y_test == 0) & (pred == 0)).sum())

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f2 = (
            (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
            if (beta**2 * precision + recall) > 0 else 0.0
        )

        # Transition recall
        n_trans = int(is_transition.sum())
        if n_trans > 0:
            trans_caught = int((is_transition & (pred == 1)).sum())
            trans_recall = trans_caught / n_trans
        else:
            trans_caught = 0
            trans_recall = np.nan

        rows.append({
            "prob_threshold": thresh,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "f2": f2,
            "transition_recall": trans_recall,
            "transitions_caught": trans_caught,
            "n_transitions": n_trans,
            "tp": tp, "fn": fn, "fp": fp,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Visualization
# =============================================================================


def plot_transition_threshold_sweep(sweep_df: pd.DataFrame, output_path: str):
    """Plot threshold sweep with transition-specific metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: overall metrics
    ax1.plot(sweep_df["pred_threshold"], sweep_df["overall_recall"], "o-", label="Overall Recall", linewidth=2)
    ax1.plot(sweep_df["pred_threshold"], sweep_df["overall_precision"], "s-", label="Overall Precision", linewidth=2)
    ax1.plot(sweep_df["pred_threshold"], sweep_df["overall_f2"], "D-", label="Overall F2", linewidth=2, color="red")
    ax1.axvline(x=20, color="gray", linestyle="--", alpha=0.5, label="Default (20)")

    best_idx = sweep_df["overall_f2"].idxmax()
    best_thresh = sweep_df.loc[best_idx, "pred_threshold"]
    ax1.axvline(x=best_thresh, color="red", linestyle=":", alpha=0.7, label=f"Best F2 ({best_thresh})")

    ax1.set_xlabel("Prediction Threshold (µg/g)")
    ax1.set_ylabel("Score")
    ax1.set_title("Overall Spike Detection")
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    # Right: transition metrics
    ax2.plot(sweep_df["pred_threshold"], sweep_df["transition_recall"], "o-",
             label="Transition Recall", linewidth=2, color="darkred")
    ax2.plot(sweep_df["pred_threshold"], sweep_df["transition_f2"], "D-",
             label="Transition F2", linewidth=2, color="orange")
    ax2.plot(sweep_df["pred_threshold"], sweep_df["overall_f2"], "^--",
             label="Overall F2 (reference)", linewidth=1.5, color="gray", alpha=0.6)
    ax2.axvline(x=20, color="gray", linestyle="--", alpha=0.5, label="Default (20)")

    best_trans_idx = sweep_df["transition_f2"].idxmax()
    best_trans_thresh = sweep_df.loc[best_trans_idx, "pred_threshold"]
    ax2.axvline(x=best_trans_thresh, color="darkred", linestyle=":", alpha=0.7,
                label=f"Best Trans F2 ({best_trans_thresh})")

    ax2.set_xlabel("Prediction Threshold (µg/g)")
    ax2.set_ylabel("Score")
    ax2.set_title("Spike TRANSITION Detection\n(below 20 → above 20)")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved transition threshold sweep to {output_path}")


def plot_comparison(
    sweep_df: pd.DataFrame,
    clf_sweep_df: pd.DataFrame,
    naive_trans_recall: float,
    output_path: str,
):
    """Compare: ensemble@threshold vs classifier vs naive on transition recall."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Ensemble threshold sweep
    ax.plot(
        sweep_df["pred_threshold"],
        sweep_df["transition_recall"],
        "o-", label="Ensemble (threshold sweep)", linewidth=2, color="steelblue",
    )

    # Classifier probability sweep (map prob thresholds to a comparable x-axis)
    # Show as separate series on secondary axis
    ax2 = ax.twiny()
    ax2.plot(
        clf_sweep_df["prob_threshold"],
        clf_sweep_df["transition_recall"],
        "s-", label="Spike Classifier (prob sweep)", linewidth=2, color="forestgreen",
    )
    ax2.set_xlabel("Classifier Probability Threshold", color="forestgreen")

    # Naive baseline
    ax.axhline(y=naive_trans_recall, color="red", linestyle="--", linewidth=2,
               label=f"Naive (transition recall = {naive_trans_recall:.3f})")

    ax.set_xlabel("Ensemble Prediction Threshold (µg/g)")
    ax.set_ylabel("Transition Recall")
    ax.set_title("Spike Transition Detection: Model Comparison")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {output_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Spike Transition Analysis")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed (e.g., 123 for independent test set)")
    args = parser.parse_args()

    # Override seed before anything runs
    if args.seed is not None:
        config.RANDOM_SEED = args.seed
        print(f"Using seed={args.seed} (independent test set)")

    seed_suffix = f"_seed{config.RANDOM_SEED}" if config.RANDOM_SEED != 42 else ""

    global OUTPUT_DIR, RETRO_CACHE
    OUTPUT_DIR = os.path.join("eval_results", f"spike_transition{seed_suffix}")
    RETRO_CACHE = os.path.join("eval_results", f"retro{seed_suffix}", "retro_regression_xgb.parquet")

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load raw DA
    print("Loading raw DA measurements...")
    raw_da = load_raw_da_measurements()
    raw_da["date"] = pd.to_datetime(raw_da["date"])

    # Load cached retrospective results
    print(f"Loading cached results from {RETRO_CACHE}...")
    if not os.path.exists(RETRO_CACHE):
        print(f"ERROR: {RETRO_CACHE} not found.")
        print(f"Run: python3 spike_detection_eval.py --seed {config.RANDOM_SEED}")
        sys.exit(1)

    retro = pd.read_parquet(RETRO_CACHE)
    retro["date"] = pd.to_datetime(retro["date"])
    print(f"  Loaded {len(retro)} retrospective results")

    # Identify transitions
    print("Identifying spike transitions...")
    retro = identify_transitions(retro, raw_da)
    n_trans = retro["is_spike_transition"].sum()
    n_spikes = retro["is_actual_spike"].sum()
    print(f"  {n_spikes} spike events, {n_trans} transitions in test set")

    # -------------------------------------------------------------------------
    # Part 1: Transition-specific threshold sweep for ensemble
    # -------------------------------------------------------------------------
    print("\n=== Part 1: Transition-Specific Threshold Sweep ===")

    # Sweep for ensemble
    if "ensemble_prediction" in retro.columns:
        ens_sweep = transition_threshold_sweep(retro, "ensemble_prediction")
        ens_sweep.to_csv(os.path.join(OUTPUT_DIR, "transition_threshold_sweep_ensemble.csv"), index=False)
        plot_transition_threshold_sweep(ens_sweep, os.path.join(OUTPUT_DIR, "transition_threshold_sweep.png"))

        print("\nEnsemble transition threshold sweep:")
        print(f"{'Threshold':<10} {'Trans Recall':<14} {'Trans F2':<10} {'Overall F2':<10} {'Trans Caught'}")
        print("-" * 60)
        for _, row in ens_sweep.iterrows():
            print(
                f"{row['pred_threshold']:<10.0f} "
                f"{row['transition_recall']:<14.3f} "
                f"{row['transition_f2']:<10.3f} "
                f"{row['overall_f2']:<10.3f} "
                f"{int(row['transitions_caught'])}/{int(row['n_transitions'])}"
            )

        best_trans = ens_sweep.loc[ens_sweep["transition_f2"].idxmax()]
        print(f"\nBest transition F2: {best_trans['transition_f2']:.3f} at threshold {best_trans['pred_threshold']}")
        print(f"  Transition recall: {best_trans['transition_recall']:.3f}")
        print(f"  Overall F2: {best_trans['overall_f2']:.3f}")

    # Naive transition recall
    naive_trans = retro[retro["is_spike_transition"]]
    naive_trans_recall = (naive_trans["naive_prediction"] >= SPIKE_THRESHOLD).mean() if len(naive_trans) > 0 else 0
    print(f"\nNaive transition recall: {naive_trans_recall:.3f}")

    # XGB transition recall
    if "predicted_da" in retro.columns:
        xgb_trans_recall = (naive_trans["predicted_da"] >= SPIKE_THRESHOLD).mean() if len(naive_trans) > 0 else 0
        print(f"XGB transition recall (at 20): {xgb_trans_recall:.3f}")

    # -------------------------------------------------------------------------
    # Part 2: Prototype Binary Spike Classifier
    # -------------------------------------------------------------------------
    print("\n=== Part 2: Prototype Binary Spike Classifier ===")

    # Build feature frame
    feature_frame = build_feature_frame_for_classifier()

    # Train and evaluate classifier
    print("Training spike classifier...")
    test_points = retro[["date", "site", "actual_da"]].copy()
    clf_results = train_spike_classifier(feature_frame, test_points)

    if clf_results:
        test_df = clf_results["test_df"]
        proba = clf_results["proba"]
        y_test = clf_results["y_test"]

        test_df["date"] = pd.to_datetime(test_df["date"])
        # Transition = spike AND previous observation was below threshold.
        # Use da_raw_prev_obs_1 (the previous observation lag, always leak-free).
        prev_feat = "da_raw_prev_obs_1" if "da_raw_prev_obs_1" in test_df.columns else "last_observed_da_raw"
        is_transition = (
            (test_df["spike_target"].values == 1)
            & (test_df[prev_feat].fillna(SPIKE_THRESHOLD + 1).values < SPIKE_THRESHOLD)
        )

        # Evaluate at different probability thresholds
        clf_sweep = evaluate_classifier_at_thresholds(proba, y_test, is_transition)
        clf_sweep.to_csv(os.path.join(OUTPUT_DIR, "classifier_prob_sweep.csv"), index=False)

        print("\nClassifier probability threshold sweep:")
        print(f"{'Prob Thresh':<12} {'Recall':<8} {'Precision':<10} {'F2':<8} {'Trans Recall':<13} {'Trans Caught'}")
        print("-" * 65)
        for _, row in clf_sweep.iterrows():
            tr = row["transition_recall"]
            tr_str = f"{tr:.3f}" if pd.notna(tr) else "N/A"
            print(
                f"{row['prob_threshold']:<12.2f} "
                f"{row['recall']:<8.3f} "
                f"{row['precision']:<10.3f} "
                f"{row['f2']:<8.3f} "
                f"{tr_str:<13} "
                f"{int(row['transitions_caught'])}/{int(row['n_transitions'])}"
            )

        best_clf = clf_sweep.loc[clf_sweep["f2"].idxmax()]
        print(f"\nBest full classifier F2: {best_clf['f2']:.3f} at prob threshold {best_clf['prob_threshold']:.2f}")
        print(f"  Transition recall: {best_clf['transition_recall']:.3f}")

        # Top features (full classifier)
        print("\nTop 15 features for full spike classifier:")
        for feat, imp in clf_results["top_features"]:
            print(f"  {feat:<35} {imp:.4f}")

        # --- Transition-focused classifier (safe-baseline only) ---
        proba_trans = clf_results["proba_trans"]
        valid_trans = ~np.isnan(proba_trans)
        if valid_trans.sum() > 0:
            trans_sweep = evaluate_classifier_at_thresholds(
                proba_trans[valid_trans],
                y_test[valid_trans],
                is_transition[valid_trans],
            )
            trans_sweep.to_csv(os.path.join(OUTPUT_DIR, "transition_classifier_prob_sweep.csv"), index=False)

            print("\nTransition-focused classifier (trained only on safe-baseline rows):")
            print(f"{'Prob Thresh':<12} {'Recall':<8} {'Precision':<10} {'F2':<8} {'Trans Recall':<13} {'Trans Caught'}")
            print("-" * 65)
            for _, row in trans_sweep.iterrows():
                tr = row["transition_recall"]
                tr_str = f"{tr:.3f}" if pd.notna(tr) else "N/A"
                print(
                    f"{row['prob_threshold']:<12.2f} "
                    f"{row['recall']:<8.3f} "
                    f"{row['precision']:<10.3f} "
                    f"{row['f2']:<8.3f} "
                    f"{tr_str:<13} "
                    f"{int(row['transitions_caught'])}/{int(row['n_transitions'])}"
                )

            best_trans_clf = trans_sweep.loc[trans_sweep["f2"].idxmax()]
            print(f"\nBest transition-focused F2: {best_trans_clf['f2']:.3f} "
                  f"at prob {best_trans_clf['prob_threshold']:.2f}")
            print(f"  Transition recall: {best_trans_clf['transition_recall']:.3f}")

            print("\nTop 15 features for transition-focused classifier:")
            for feat, imp in clf_results["top_features_trans"]:
                print(f"  {feat:<35} {imp:.4f}")

        # Comparison plot
        if "ensemble_prediction" in retro.columns:
            plot_comparison(ens_sweep, clf_sweep, naive_trans_recall,
                           os.path.join(OUTPUT_DIR, "comparison_plot.png"))

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SPIKE TRANSITION ANALYSIS — SUMMARY")
    print("=" * 80)

    if "ensemble_prediction" in retro.columns:
        print(f"\nEnsemble at threshold 20: transition recall = {ens_sweep[ens_sweep['pred_threshold'] == 20]['transition_recall'].values[0]:.3f}")
        best_t = ens_sweep.loc[ens_sweep["transition_f2"].idxmax()]
        print(f"Ensemble at threshold {best_t['pred_threshold']:.0f}: transition recall = {best_t['transition_recall']:.3f} (best transition F2)")
    print(f"Naive: transition recall = {naive_trans_recall:.3f}")

    if clf_results:
        best_c = clf_sweep.loc[clf_sweep["f2"].idxmax()]
        print(f"Full classifier at prob {best_c['prob_threshold']:.2f}: transition recall = {best_c['transition_recall']:.3f}")
        proba_trans = clf_results["proba_trans"]
        valid_trans = ~np.isnan(proba_trans)
        if valid_trans.sum() > 0 and "trans_sweep" in dir():
            best_tc = trans_sweep.loc[trans_sweep["transition_recall"].idxmax()]
            print(f"Transition-focused classifier at prob {best_tc['prob_threshold']:.2f}: "
                  f"transition recall = {best_tc['transition_recall']:.3f} "
                  f"(only evaluated on safe-baseline test rows)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

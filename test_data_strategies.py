"""
Data Sparsity Strategy Comparison Experiment
=============================================

Compares 7 training data strategies for the DATect DA forecasting system,
all evaluated on the same shared test set of real DA measurements.

Usage:
    python3 test_data_strategies.py

Output:
    - Console comparison table (per-site and overall R², MAE, Spike F1)
    - data_strategy_results.json
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, f1_score
from tqdm import tqdm

import config
from forecasting.raw_data_forecaster import (
    load_raw_da_measurements,
    aggregate_raw_to_weekly,
    build_raw_feature_frame,
    get_site_training_frame,
    get_site_anchor_row,
    recompute_test_row_persistence_features,
    get_last_known_raw_da,
)
from forecasting.raw_model_factory import build_xgb_regressor, build_rf_regressor
from forecasting.feature_utils import add_temporal_features, create_transformer
from forecasting.per_site_models import (
    apply_site_xgb_params,
    apply_site_rf_params,
    get_site_ensemble_weights,
    get_site_clip_params,
    compute_site_drop_cols,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SITE_NEIGHBORS = {
    "Kalaloch": ["Quinault"],
    "Quinault": ["Kalaloch", "Copalis"],
    "Copalis": ["Quinault", "Twin Harbors"],
    "Twin Harbors": ["Copalis", "Long Beach"],
    "Long Beach": ["Twin Harbors", "Clatsop Beach"],
    "Clatsop Beach": ["Long Beach", "Cannon Beach"],
    "Cannon Beach": ["Clatsop Beach"],
    "Newport": [],
    "Coos Bay": [],
    "Gold Beach": [],
}

SPARSE_THRESHOLD = 80  # Strategy 7: naive-only below this
INTERP_WEIGHT = 0.3  # Strategy 6: weight for interpolated rows
MIN_TEST_DATE = "2008-01-01"
MAX_TEST_PER_SITE = None  # No cap — use full 20% per site, matching production retrospective
OUTPUT_FILE = "data_strategy_results.json"

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_data():
    """Load and prepare all data needed by all strategies.

    Returns (feature_frame, raw_df, processed_df) where:
      - feature_frame: env features + raw DA + lags + derived features + da_interpolated
      - raw_df: raw DA measurements from CSVs (date, site, da_raw)
      - processed_df: full parquet with interpolated 'da' column
    """
    print("Loading data...")
    processed_df = pd.read_parquet(config.FINAL_OUTPUT_PATH)
    processed_df["date"] = pd.to_datetime(processed_df["date"])
    processed_df = processed_df.sort_values(["site", "date"]).reset_index(drop=True)

    raw_df = load_raw_da_measurements()

    processed_with_temporal = add_temporal_features(processed_df.copy())
    raw_weekly = aggregate_raw_to_weekly(raw_df)
    feature_frame = build_raw_feature_frame(processed_with_temporal, raw_weekly)

    # Merge interpolated DA into feature_frame as 'da_interpolated'
    interp_cols = processed_df[["site", "date", "da"]].rename(
        columns={"da": "da_interpolated"}
    )
    feature_frame = feature_frame.merge(interp_cols, on=["site", "date"], how="left")

    print(f"  Feature frame: {len(feature_frame)} rows, {len(feature_frame.columns)} cols")
    print(f"  Raw measurements: {len(raw_df)} total")
    print(f"  Sites: {sorted(feature_frame['site'].unique())}")
    return feature_frame, raw_df, processed_df


# ---------------------------------------------------------------------------
# Test Point Sampling (mirrors run_retrospective_evaluation lines 642-696)
# ---------------------------------------------------------------------------


def sample_test_points(raw_df: pd.DataFrame, feature_frame: pd.DataFrame) -> list[dict]:
    """Sample ~20% of raw observations per site as test points.

    Applies history requirement filter matching the production retrospective.
    """
    print("Sampling test points...")
    min_test_ts = pd.Timestamp(MIN_TEST_DATE)
    forecast_horizon = config.FORECAST_HORIZON_DAYS
    min_training = getattr(config, "MIN_TRAINING_SAMPLES", 10)
    history_frac = getattr(config, "HISTORY_REQUIREMENT_FRACTION", 0.33)

    candidate_raw = raw_df[raw_df["date"] >= min_test_ts].copy()
    site_total_counts = raw_df.groupby("site")["date"].size().to_dict()

    valid_rows = []
    for _, row in candidate_raw.iterrows():
        anchor_dt = row["date"] - pd.Timedelta(days=forecast_horizon)
        site = row["site"]
        total_site = site_total_counts.get(site, 0)
        if total_site == 0:
            continue
        min_required = max(int(np.ceil(history_frac * total_site)), min_training)
        n_history = len(
            raw_df[(raw_df["site"] == site) & (raw_df["date"] <= anchor_dt)]
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

    valid_df = pd.DataFrame(valid_rows)

    rng = np.random.RandomState(config.RANDOM_SEED)
    sampled_rows = []
    for site, site_df in valid_df.groupby("site"):
        site_df = site_df.sort_values("date")
        n_candidates = len(site_df)
        total_site = site_total_counts.get(site, n_candidates)
        target = min(int(np.ceil(0.2 * total_site)), n_candidates)
        if MAX_TEST_PER_SITE is not None:
            target = min(target, MAX_TEST_PER_SITE)
        if target <= 0:
            continue
        idx = rng.choice(n_candidates, size=target, replace=False)
        sampled_rows.append(site_df.iloc[idx])

    test_samples = pd.concat(sampled_rows, ignore_index=True)
    test_points = [
        {"date": row["date"], "site": row["site"], "da_raw": row["da_raw"]}
        for _, row in test_samples.iterrows()
    ]

    print(f"  Total test points: {len(test_points)}")
    for site in sorted(test_samples["site"].unique()):
        n = len(test_samples[test_samples["site"] == site])
        print(f"    {site}: {n} test points")
    return test_points


# ---------------------------------------------------------------------------
# Core Prediction (mirrors _run_single_raw_validation lines 917-1100)
# ---------------------------------------------------------------------------


def predict_point(
    test_point: dict,
    train_data: pd.DataFrame,
    feature_frame: pd.DataFrame,
    use_per_site: bool = True,
    sample_weights: Optional[np.ndarray] = None,
    extra_train_features: Optional[pd.DataFrame] = None,
    extra_test_features: Optional[dict] = None,
    real_obs_train: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """Run XGB + RF + Naive ensemble prediction for a single test point.

    Parameters
    ----------
    test_point : dict with date, site, da_raw
    train_data : DataFrame with da_raw column as training target
    feature_frame : full feature frame for anchor row lookup
    use_per_site : whether to apply per-site feature drops and ensemble weights
    sample_weights : optional weights for XGB/RF fit
    extra_train_features : extra columns to add to training features
    extra_test_features : extra columns to add to test features (dict of col->value)
    real_obs_train : real-observation-only subset for persistence recomputation
                     (used when train_data contains interpolated rows)
    """
    test_date = pd.Timestamp(test_point["date"])
    site = test_point["site"]
    actual_da = float(test_point["da_raw"])

    anchor_date = test_date - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)

    # Get test row (env features from anchor date)
    test_row = get_site_anchor_row(
        feature_frame, site, test_date, anchor_date, max_date_diff_days=28
    )
    if test_row is None:
        return None

    # Recompute persistence features from real observations only
    persistence_source = real_obs_train if real_obs_train is not None else train_data
    # Filter persistence source to only real observations
    if "da_raw" in persistence_source.columns:
        persistence_clean = persistence_source.dropna(subset=["da_raw"])
        if not persistence_clean.empty:
            test_row = recompute_test_row_persistence_features(
                test_row, persistence_clean, config.SPIKE_THRESHOLD
            )

    # Add temporal features
    train_data = add_temporal_features(train_data.copy())
    test_row = add_temporal_features(test_row)

    # Feature preparation
    zero_imp = list(getattr(config, "ZERO_IMPORTANCE_FEATURES", []))
    drop_cols = ["date", "site", "da_raw", "da", "da_interpolated"] + zero_imp

    if use_per_site:
        drop_cols = compute_site_drop_cols(
            drop_cols, train_data.columns.tolist(), site
        )

    # Add extra features to train and test if provided
    if extra_train_features is not None:
        for col in extra_train_features.columns:
            if col not in train_data.columns:
                train_data[col] = extra_train_features[col].values

    try:
        transformer, X_train = create_transformer(train_data, drop_cols)
        y_train = train_data["da_raw"].astype(float).copy()

        X_train_processed = transformer.fit_transform(X_train)

        X_test = test_row.drop(columns=drop_cols, errors="ignore")
        # Add extra test features
        if extra_test_features:
            for col, val in extra_test_features.items():
                X_test[col] = val
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        X_test_processed = transformer.transform(X_test)
    except Exception:
        return None

    # Post-processing: clip predictions
    clip_q = getattr(config, "PREDICTION_CLIP_Q", 0.99)

    def _postprocess(value: float) -> float:
        value = max(0.0, value)
        if use_per_site:
            sq, sm = get_site_clip_params(site)
            cq = sq if sq is not None else clip_q
        else:
            cq = clip_q
            sm = None
        if cq is not None and len(y_train) > 0:
            clip_max = float(np.quantile(y_train, min(cq, 1.0)))
            value = min(value, clip_max)
        if sm is not None:
            value = min(value, sm)
        return float(value)

    # XGBoost
    xgb_params = dict(config.XGB_REGRESSION_PARAMS)
    xgb_params["n_jobs"] = 1
    if use_per_site:
        xgb_params = apply_site_xgb_params(xgb_params, site)
    xgb_model = build_xgb_regressor(xgb_params)
    xgb_model.fit(X_train_processed, y_train, sample_weight=sample_weights)
    xgb_pred = _postprocess(float(xgb_model.predict(X_test_processed)[0]))

    # Random Forest
    rf_params = dict(getattr(config, "RF_REGRESSION_PARAMS", {}))
    if use_per_site:
        rf_params = apply_site_rf_params(rf_params, site)
    rf_model = build_rf_regressor(rf_params)
    rf_model.fit(X_train_processed, y_train, sample_weight=sample_weights)
    rf_pred = _postprocess(float(rf_model.predict(X_test_processed)[0]))

    # Naive (always from real observations)
    naive_source = real_obs_train if real_obs_train is not None else train_data
    naive_pred = get_last_known_raw_da(
        naive_source, anchor_date=anchor_date,
        max_age_days=getattr(config, "PERSISTENCE_MAX_DAYS", None),
    )
    if naive_pred is None:
        naive_pred = 0.0

    # Ensemble blend
    if use_per_site:
        w_xgb, w_rf, w_naive = get_site_ensemble_weights(site)
    else:
        w_xgb, w_rf, w_naive = 0.30, 0.50, 0.20
    ensemble_pred = w_xgb * xgb_pred + w_rf * rf_pred + w_naive * naive_pred

    return {
        "site": site,
        "date": str(test_date.date()),
        "actual_da": actual_da,
        "predicted_da": ensemble_pred,
        "xgb_pred": xgb_pred,
        "rf_pred": rf_pred,
        "naive_pred": naive_pred,
        "training_samples": len(train_data),
    }


# ---------------------------------------------------------------------------
# Strategy Functions
# ---------------------------------------------------------------------------


def strategy_baseline(site, anchor_date, feature_frame, **kw):
    """Strategy 1: Train on raw DA only (current system)."""
    min_samples = getattr(config, "MIN_TRAINING_SAMPLES", 10)
    train_data = get_site_training_frame(feature_frame, site, anchor_date, min_samples)
    if train_data is None:
        return None, None, {}
    return train_data, None, {}


def strategy_interpolated(site, anchor_date, feature_frame, **kw):
    """Strategy 2: Train on interpolated DA (all rows for site)."""
    site_data = feature_frame[feature_frame["site"] == site].copy()
    site_data = site_data[site_data["date"] <= anchor_date].copy()
    if len(site_data) < 5:
        return None, None, {}

    # Use da_raw where available, fall back to da_interpolated
    mask_real = site_data["da_raw"].notna()
    site_data.loc[~mask_real, "da_raw"] = site_data.loc[~mask_real, "da_interpolated"]

    # Drop rows where neither real nor interpolated DA is available
    site_data = site_data.dropna(subset=["da_raw"])
    if len(site_data) < 5:
        return None, None, {}

    # Keep track of real observations for persistence recomputation
    real_obs = site_data[mask_real[site_data.index]].copy()

    return site_data, None, {"real_obs_train": real_obs}


def strategy_pooled(site, anchor_date, feature_frame, **kw):
    """Strategy 3: Multi-site pooled global model with site encoding."""
    min_samples = getattr(config, "MIN_TRAINING_SAMPLES", 10)
    all_sites = sorted(feature_frame["site"].unique())
    site_to_code = {s: float(i) for i, s in enumerate(all_sites)}

    frames = []
    for s in all_sites:
        s_data = get_site_training_frame(feature_frame, s, anchor_date, min_training_samples=1)
        if s_data is not None and len(s_data) > 0:
            frames.append(s_data)

    if not frames:
        return None, None, {}

    pooled = pd.concat(frames, ignore_index=True)
    pooled["site_encoded"] = pooled["site"].map(site_to_code)

    if len(pooled) < min_samples:
        return None, None, {}

    return pooled, None, {
        "use_per_site": False,
        "extra_test_features": {"site_encoded": site_to_code.get(site, 0.0)},
    }


def strategy_pretrain_finetune(site, anchor_date, feature_frame, **kw):
    """Strategy 4: Global pretrain + per-site fine-tune via stacking."""
    min_samples = getattr(config, "MIN_TRAINING_SAMPLES", 10)
    all_sites = sorted(feature_frame["site"].unique())
    site_to_code = {s: float(i) for i, s in enumerate(all_sites)}

    # Phase 1: Build pooled training data and train global model
    frames = []
    for s in all_sites:
        s_data = get_site_training_frame(feature_frame, s, anchor_date, min_training_samples=1)
        if s_data is not None and len(s_data) > 0:
            frames.append(s_data)

    if not frames:
        return None, None, {}

    pooled = pd.concat(frames, ignore_index=True)
    pooled["site_encoded"] = pooled["site"].map(site_to_code)

    # Train global XGB on pooled data
    zero_imp = list(getattr(config, "ZERO_IMPORTANCE_FEATURES", []))
    global_drop = ["date", "site", "da_raw", "da", "da_interpolated"] + zero_imp
    pooled_with_temporal = add_temporal_features(pooled.copy())

    try:
        global_transformer, X_global = create_transformer(pooled_with_temporal, global_drop)
        y_global = pooled_with_temporal["da_raw"].astype(float)
        X_global_proc = global_transformer.fit_transform(X_global)

        global_xgb = build_xgb_regressor({**config.XGB_REGRESSION_PARAMS, "n_jobs": 1})
        global_xgb.fit(X_global_proc, y_global)
    except Exception:
        return None, None, {}

    # Phase 2: Get per-site data and add global predictions as feature
    site_train = get_site_training_frame(feature_frame, site, anchor_date, min_samples)
    if site_train is None:
        return None, None, {}

    site_train = site_train.copy()
    site_train["site_encoded"] = site_to_code.get(site, 0.0)
    site_train_temporal = add_temporal_features(site_train.copy())

    try:
        X_site_for_global = site_train_temporal.drop(columns=global_drop, errors="ignore")
        X_site_for_global = X_site_for_global.reindex(
            columns=X_global.columns, fill_value=0
        )
        X_site_proc = global_transformer.transform(X_site_for_global)
        global_preds = global_xgb.predict(X_site_proc)
        site_train["global_pred"] = global_preds.clip(min=0)
    except Exception:
        site_train["global_pred"] = 0.0

    return site_train, None, {
        "extra_test_features": {
            "site_encoded": site_to_code.get(site, 0.0),
            "global_pred": None,  # placeholder, computed below
        },
        "_global_model": (global_xgb, global_transformer, X_global.columns.tolist()),
        "_global_drop": global_drop,
        "_site_to_code": site_to_code,
    }


def strategy_neighbor_features(site, anchor_date, feature_frame, raw_df=None, **kw):
    """Strategy 5: Baseline + neighbor-site last DA as features."""
    min_samples = getattr(config, "MIN_TRAINING_SAMPLES", 10)
    train_data = get_site_training_frame(feature_frame, site, anchor_date, min_samples)
    if train_data is None:
        return None, None, {}

    neighbors = SITE_NEIGHBORS.get(site, [])
    if not neighbors or raw_df is None:
        return train_data, None, {}

    train_data = train_data.copy()
    extra_test = {}

    for neighbor in neighbors:
        col_name = f"neighbor_{neighbor.replace(' ', '_')}_da"
        neighbor_raw = raw_df[raw_df["site"] == neighbor].sort_values("date")
        if neighbor_raw.empty:
            train_data[col_name] = np.nan
            extra_test[col_name] = np.nan
            continue

        # For each training row, find most recent neighbor DA before that row's date
        neighbor_dates = neighbor_raw["date"].values
        neighbor_vals = neighbor_raw["da_raw"].values

        neighbor_da_values = []
        for _, row in train_data.iterrows():
            mask = neighbor_dates <= row["date"]
            if mask.any():
                neighbor_da_values.append(float(neighbor_vals[mask][-1]))
            else:
                neighbor_da_values.append(np.nan)
        train_data[col_name] = neighbor_da_values

        # For test row: most recent neighbor DA before anchor_date
        mask = neighbor_dates <= anchor_date
        if mask.any():
            extra_test[col_name] = float(neighbor_vals[mask][-1])
        else:
            extra_test[col_name] = np.nan

    return train_data, None, {"extra_test_features": extra_test}


def strategy_weighted_interpolated(site, anchor_date, feature_frame, **kw):
    """Strategy 6: Train on all rows, real=1.0, interpolated=0.3 weight."""
    site_data = feature_frame[feature_frame["site"] == site].copy()
    site_data = site_data[site_data["date"] <= anchor_date].copy()
    if len(site_data) < 5:
        return None, None, {}

    # Track which rows are real vs interpolated
    mask_real = site_data["da_raw"].notna()
    site_data.loc[~mask_real, "da_raw"] = site_data.loc[~mask_real, "da_interpolated"]
    site_data = site_data.dropna(subset=["da_raw"])
    if len(site_data) < 5:
        return None, None, {}

    # Build sample weights
    weights = np.where(mask_real[site_data.index], 1.0, INTERP_WEIGHT)

    real_obs = site_data[mask_real[site_data.index]].copy()

    return site_data, weights, {"real_obs_train": real_obs}


def strategy_naive_sparse(site, anchor_date, feature_frame, raw_df=None, **kw):
    """Strategy 7: Naive-only for sites with <80 observations, else baseline."""
    if raw_df is not None:
        site_obs = len(raw_df[raw_df["site"] == site])
        if site_obs < SPARSE_THRESHOLD:
            return None, None, {"naive_only": True}

    return strategy_baseline(site, anchor_date, feature_frame)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


STRATEGIES = {
    "1_baseline": (strategy_baseline, "Train on raw DA only (current system)"),
    "2_interpolated": (strategy_interpolated, "Train on interpolated DA (all rows)"),
    "3_pooled": (strategy_pooled, "Multi-site pooled global model"),
    "4_pretrain_finetune": (strategy_pretrain_finetune, "Global pretrain + per-site fine-tune"),
    "5_neighbor_features": (strategy_neighbor_features, "Add neighbor-site DA as features"),
    "6_weighted_interp": (strategy_weighted_interpolated, "Weighted interpolated (real=1.0, interp=0.3)"),
    "7_naive_sparse": (strategy_naive_sparse, "Naive-only for sparse sites (<80 obs)"),
}


def evaluate_strategy(
    name: str,
    strategy_fn,
    test_points: list[dict],
    feature_frame: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate a strategy across all test points."""
    results = []

    for tp in tqdm(test_points, desc=f"  {name}", unit="pt"):
        site = tp["site"]
        test_date = pd.Timestamp(tp["date"])
        anchor_date = test_date - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)

        # Get training data from strategy
        train_data, weights, extras = strategy_fn(
            site, anchor_date, feature_frame, raw_df=raw_df
        )

        use_per_site = extras.get("use_per_site", True)
        extra_test = extras.get("extra_test_features", None)
        real_obs = extras.get("real_obs_train", None)
        naive_only = extras.get("naive_only", False)

        if naive_only:
            # Just use naive prediction
            naive_source = get_site_training_frame(
                feature_frame, site, anchor_date,
                getattr(config, "MIN_TRAINING_SAMPLES", 10)
            )
            if naive_source is None:
                continue
            naive_pred = get_last_known_raw_da(naive_source, anchor_date=anchor_date)
            if naive_pred is None:
                continue
            results.append({
                "site": site,
                "date": str(test_date.date()),
                "actual_da": float(tp["da_raw"]),
                "predicted_da": naive_pred,
                "xgb_pred": naive_pred,
                "rf_pred": naive_pred,
                "naive_pred": naive_pred,
                "training_samples": len(naive_source),
            })
            continue

        if train_data is None:
            continue

        # Strategy 4 special handling: compute global_pred for test row
        if "_global_model" in extras and extra_test and "global_pred" in extra_test:
            global_xgb, global_transformer, global_cols = extras["_global_model"]
            global_drop = extras["_global_drop"]
            site_to_code = extras["_site_to_code"]

            test_row_for_global = get_site_anchor_row(
                feature_frame, site, test_date, anchor_date, max_date_diff_days=28
            )
            if test_row_for_global is not None:
                test_row_for_global = add_temporal_features(test_row_for_global)
                test_row_for_global["site_encoded"] = site_to_code.get(site, 0.0)
                X_tg = test_row_for_global.drop(columns=global_drop, errors="ignore")
                X_tg = X_tg.reindex(columns=global_cols, fill_value=0)
                try:
                    global_pred_val = float(
                        global_xgb.predict(global_transformer.transform(X_tg))[0]
                    )
                    extra_test["global_pred"] = max(0.0, global_pred_val)
                except Exception:
                    extra_test["global_pred"] = 0.0
            else:
                extra_test["global_pred"] = 0.0

        result = predict_point(
            tp, train_data, feature_frame,
            use_per_site=use_per_site,
            sample_weights=weights,
            extra_test_features=extra_test,
            real_obs_train=real_obs,
        )
        if result is not None:
            results.append(result)

    return pd.DataFrame(results) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _safe_metrics(actual, predicted, spike_t):
    """Compute R², MAE, Spike F1 for a pair of arrays."""
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    if valid.sum() < 2:
        return {"r2": None, "mae": None, "spike_f1": None}
    a, p = actual[valid], predicted[valid]
    return {
        "r2": round(float(r2_score(a, p)), 4),
        "mae": round(float(mean_absolute_error(a, p)), 2),
        "spike_f1": round(float(f1_score(
            (a > spike_t).astype(int),
            (p > spike_t).astype(int),
            zero_division=0,
        )), 4),
    }


def compute_metrics(results_df: pd.DataFrame) -> dict:
    """Compute R², MAE, Spike F1 for ensemble + individual models."""
    if results_df.empty or len(results_df) < 2:
        return {
            "r2": None, "mae": None, "spike_f1": None, "n": 0,
            "xgb_r2": None, "rf_r2": None, "naive_r2": None,
        }

    actual = results_df["actual_da"].values
    spike_t = config.SPIKE_THRESHOLD

    # Ensemble metrics
    ens = _safe_metrics(actual, results_df["predicted_da"].values, spike_t)

    # Per-model R² (for comparing which model benefits most from each strategy)
    xgb_m = _safe_metrics(actual, results_df["xgb_pred"].values, spike_t)
    rf_m = _safe_metrics(actual, results_df["rf_pred"].values, spike_t)
    naive_m = _safe_metrics(actual, results_df["naive_pred"].values, spike_t)

    return {
        **ens,
        "n": len(results_df),
        "xgb_r2": xgb_m["r2"],
        "xgb_mae": xgb_m["mae"],
        "rf_r2": rf_m["r2"],
        "rf_mae": rf_m["mae"],
        "naive_r2": naive_m["r2"],
        "naive_mae": naive_m["mae"],
    }


def compute_all_metrics(results_df: pd.DataFrame) -> dict:
    """Compute overall + per-site metrics."""
    overall = compute_metrics(results_df)
    per_site = {}
    if not results_df.empty:
        for site in sorted(results_df["site"].unique()):
            site_df = results_df[results_df["site"] == site]
            per_site[site] = compute_metrics(site_df)
    return {"overall": overall, "per_site": per_site}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_comparison_table(all_metrics: dict):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON RESULTS")
    print("=" * 100)

    # Overall table (ensemble)
    print("\n--- Overall Metrics (Ensemble) ---")
    print(f"{'Strategy':<35} {'R²':>8} {'MAE':>8} {'F1':>8} {'N':>6}")
    print("-" * 70)
    for name, metrics in all_metrics.items():
        m = metrics["overall"]
        r2 = f"{m['r2']:.4f}" if m["r2"] is not None else "N/A"
        mae = f"{m['mae']:.2f}" if m["mae"] is not None else "N/A"
        f1 = f"{m['spike_f1']:.4f}" if m["spike_f1"] is not None else "N/A"
        print(f"{name:<35} {r2:>8} {mae:>8} {f1:>8} {m['n']:>6}")

    # Per-model R² breakdown
    print(f"\n--- Per-Model R² (Overall) ---")
    print(f"{'Strategy':<35} {'Ensemble':>10} {'XGB':>10} {'RF':>10} {'Naive':>10}")
    print("-" * 80)
    for name, metrics in all_metrics.items():
        m = metrics["overall"]
        ens = f"{m['r2']:.4f}" if m["r2"] is not None else "N/A"
        xgb = f"{m.get('xgb_r2', None):.4f}" if m.get("xgb_r2") is not None else "N/A"
        rf = f"{m.get('rf_r2', None):.4f}" if m.get("rf_r2") is not None else "N/A"
        naive = f"{m.get('naive_r2', None):.4f}" if m.get("naive_r2") is not None else "N/A"
        print(f"{name:<35} {ens:>10} {xgb:>10} {rf:>10} {naive:>10}")

    # Per-model MAE breakdown
    print(f"\n--- Per-Model MAE (Overall) ---")
    print(f"{'Strategy':<35} {'Ensemble':>10} {'XGB':>10} {'RF':>10} {'Naive':>10}")
    print("-" * 80)
    for name, metrics in all_metrics.items():
        m = metrics["overall"]
        ens = f"{m['mae']:.2f}" if m["mae"] is not None else "N/A"
        xgb = f"{m.get('xgb_mae', None):.2f}" if m.get("xgb_mae") is not None else "N/A"
        rf = f"{m.get('rf_mae', None):.2f}" if m.get("rf_mae") is not None else "N/A"
        naive = f"{m.get('naive_mae', None):.2f}" if m.get("naive_mae") is not None else "N/A"
        print(f"{name:<35} {ens:>10} {xgb:>10} {rf:>10} {naive:>10}")

    # Per-site R² table
    all_sites = set()
    for metrics in all_metrics.values():
        all_sites.update(metrics["per_site"].keys())
    all_sites = sorted(all_sites)

    print(f"\n--- Per-Site R² ---")
    header = f"{'Strategy':<25}"
    for s in all_sites:
        short = s[:10]
        header += f" {short:>10}"
    print(header)
    print("-" * (25 + 11 * len(all_sites)))

    for name, metrics in all_metrics.items():
        row = f"{name:<25}"
        for s in all_sites:
            sm = metrics["per_site"].get(s, {})
            r2 = sm.get("r2")
            row += f" {r2:>10.4f}" if r2 is not None else f" {'N/A':>10}"
        print(row)

    # Per-site MAE table
    print(f"\n--- Per-Site MAE ---")
    print(header)
    print("-" * (25 + 11 * len(all_sites)))

    for name, metrics in all_metrics.items():
        row = f"{name:<25}"
        for s in all_sites:
            sm = metrics["per_site"].get(s, {})
            mae = sm.get("mae")
            row += f" {mae:>10.2f}" if mae is not None else f" {'N/A':>10}"
        print(row)

    # Per-site Spike F1 table
    print(f"\n--- Per-Site Spike F1 ---")
    print(header)
    print("-" * (25 + 11 * len(all_sites)))

    for name, metrics in all_metrics.items():
        row = f"{name:<25}"
        for s in all_sites:
            sm = metrics["per_site"].get(s, {})
            f1 = sm.get("spike_f1")
            row += f" {f1:>10.4f}" if f1 is not None else f" {'N/A':>10}"
        print(row)

    print("\n" + "=" * 100)


def save_results(all_metrics: dict, all_results: dict):
    """Save results to JSON."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "min_test_date": MIN_TEST_DATE,
            "forecast_horizon_days": config.FORECAST_HORIZON_DAYS,
            "spike_threshold": config.SPIKE_THRESHOLD,
            "sparse_threshold": SPARSE_THRESHOLD,
            "interp_weight": INTERP_WEIGHT,
            "random_seed": config.RANDOM_SEED,
        },
        "strategies": {},
    }

    for name, metrics in all_metrics.items():
        output["strategies"][name] = {
            "overall": metrics["overall"],
            "per_site": metrics["per_site"],
        }

    # Also save raw predictions for deeper analysis
    for name, df in all_results.items():
        if not df.empty:
            output["strategies"][name]["predictions"] = df.to_dict(orient="records")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("DATect Data Sparsity Strategy Comparison")
    print("=" * 60)

    feature_frame, raw_df, processed_df = load_data()
    test_points = sample_test_points(raw_df, feature_frame)

    # Show per-site observation counts
    print("\nRaw observation counts per site:")
    for site in sorted(raw_df["site"].unique()):
        n = len(raw_df[raw_df["site"] == site])
        sparse = " (SPARSE)" if n < SPARSE_THRESHOLD else ""
        print(f"  {site}: {n}{sparse}")

    all_results = {}
    all_metrics = {}

    for name, (strategy_fn, desc) in STRATEGIES.items():
        print(f"\n{'─' * 60}")
        print(f"Strategy: {name}")
        print(f"  {desc}")
        print(f"{'─' * 60}")

        results_df = evaluate_strategy(
            name, strategy_fn, test_points, feature_frame, raw_df
        )

        metrics = compute_all_metrics(results_df)
        all_results[name] = results_df
        all_metrics[name] = metrics

        m = metrics["overall"]
        r2 = f"{m['r2']:.4f}" if m["r2"] is not None else "N/A"
        mae = f"{m['mae']:.2f}" if m["mae"] is not None else "N/A"
        f1 = f"{m['spike_f1']:.4f}" if m["spike_f1"] is not None else "N/A"
        xr2 = f"{m.get('xgb_r2'):.4f}" if m.get("xgb_r2") is not None else "N/A"
        rr2 = f"{m.get('rf_r2'):.4f}" if m.get("rf_r2") is not None else "N/A"
        print(f"  Ensemble: R²={r2}, MAE={mae}, F1={f1}, N={m['n']}")
        print(f"  Per-model R²: XGB={xr2}, RF={rr2}")

    print_comparison_table(all_metrics)
    save_results(all_metrics, all_results)


if __name__ == "__main__":
    main()

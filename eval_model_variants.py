#!/usr/bin/env python3
"""
eval_model_variants.py
======================

Compares four model variants under DATect's leak-free anchor-based
retrospective protocol:

  1. xgb_standard   — XGBoost with DATect default params + per-site overrides
  2. rf_standard     — Random Forest with DATect default params + per-site overrides
  3. xgb_rf_mode     — XGBoost in Random Forest mode (XGBRFRegressor-equivalent)
  4. stacking        — Two-level stacking: XGB + RF base learners, Ridge meta-learner
                       trained on 5-fold chronological OOF predictions (leak-free)

Usage
-----
    python3 eval_model_variants.py [--seed 123] [--sample-fraction 0.40]
                                   [--sites all|WA|site_name]
                                   [--output-dir eval_results/model_variants]
                                   [--workers N]

Stacking OOF protocol (leak-free)
----------------------------------
For each test point the training set is the only data available (date <=
anchor_date).  We sort those rows by date and run TimeSeriesSplit(n_splits=5)
to generate out-of-fold predictions from XGBoost and RF.  A Ridge meta-learner
is then trained on [xgb_oof, rf_oof] targets.  At test time, both base models
are retrained on the *full* training set and their predictions are stacked as
meta-features for the Ridge model.  The OOF generation and meta-learner fitting
happen entirely within the pre-anchor window — no future data is ever touched.
"""

from __future__ import annotations

import os
# Must be set before any library imports to prevent fork+OpenMP deadlock
# in ProcessPoolExecutor workers on Linux (XGBoost/sklearn OpenMP thread pools)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, r2_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Ensure DATect project root is on sys.path so we can import config etc.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import config  # noqa: E402
from forecasting.raw_data_forecaster import (  # noqa: E402
    aggregate_raw_to_weekly,
    build_raw_feature_frame,
    get_last_known_raw_da,
    get_site_anchor_row,
    get_site_training_frame,
    load_raw_da_measurements,
    recompute_test_row_persistence_features,
)
from forecasting.feature_utils import add_temporal_features, create_transformer  # noqa: E402
from forecasting.per_site_models import (  # noqa: E402
    apply_site_rf_params,
    apply_site_xgb_params,
    compute_site_drop_cols,
    get_site_clip_params,
)
from forecasting.raw_model_factory import build_rf_regressor, build_xgb_regressor  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPIKE_THRESHOLD = config.SPIKE_THRESHOLD  # 20.0 µg/g
FORECAST_HORIZON_DAYS = config.FORECAST_HORIZON_DAYS  # 7
MIN_TRAINING_SAMPLES = max(1, int(getattr(config, "MIN_TRAINING_SAMPLES", 10)))

# XGBoost RF mode parameters (single boosting round, many parallel trees)
XGB_RF_MODE_PARAMS = {
    "n_estimators": 1,
    "num_parallel_tree": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "colsample_bynode": 0.8,   # critical: column sampling *per node* drives tree diversity
    "learning_rate": 1.0,
    "max_depth": 8,
    "reg_lambda": 1e-5,
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0,
    "device": "cpu",
}

# Minimum training rows needed to attempt stacking OOF; fall back to rf_standard
STACKING_MIN_TRAINING = 20
STACKING_OOF_SPLITS = 5

MODEL_NAMES = ["xgb_standard", "rf_standard", "xgb_rf_mode", "stacking"]


# ===========================================================================
# Helper: prediction post-processing (clip to training quantile + site max)
# ===========================================================================

def _postprocess(
    value: float,
    train_y: pd.Series,
    site: str,
    use_per_site: bool = True,
    clip_q_global: float = 0.99,
) -> float:
    value = max(0.0, float(value))
    if use_per_site:
        site_clip_q, site_clip_max = get_site_clip_params(site)
        cq = site_clip_q if site_clip_q is not None else clip_q_global
    else:
        cq = clip_q_global
        site_clip_max = None

    if cq is not None:
        clip_max = float(np.quantile(train_y, cq))
        value = min(value, clip_max)
    if site_clip_max is not None:
        value = min(value, site_clip_max)
    return value


# ===========================================================================
# Feature preparation helper (shared across all variants)
# ===========================================================================

def _prepare_features(
    train_data: pd.DataFrame,
    test_row: pd.DataFrame,
    site: str,
    use_per_site: bool = True,
) -> Tuple[np.ndarray, pd.Series, np.ndarray]:
    """
    Apply temporal features, drop zero-importance + per-site columns, impute,
    and scale.  Returns (X_train_processed, y_train, X_test_processed).
    """
    zero_imp = getattr(config, "ZERO_IMPORTANCE_FEATURES", [])
    base_drop = ["date", "site", "da_raw", "da", "_is_interpolated"] + list(zero_imp)

    if use_per_site:
        drop_cols = compute_site_drop_cols(
            base_drop, train_data.columns.tolist(), site
        )
    else:
        drop_cols = list(base_drop)

    transformer, X_train = create_transformer(train_data, drop_cols)
    y_train = train_data["da_raw"].astype(float).copy()
    X_train_processed = transformer.fit_transform(X_train)

    X_test = test_row.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_test_processed = transformer.transform(X_test)

    return X_train_processed, y_train, X_test_processed


# ===========================================================================
# Individual model predictors
# ===========================================================================

def _predict_xgb_standard(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    site: str,
    seed: int,
) -> float:
    base = dict(config.XGB_REGRESSION_PARAMS)
    base["n_jobs"] = 1
    base["random_state"] = seed
    params = apply_site_xgb_params(base, site)
    params.setdefault("nthread", 1)
    model = build_xgb_regressor(params)
    model.fit(X_train, y_train)
    return float(model.predict(X_test)[0])


def _predict_rf_standard(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    site: str,
    seed: int,
) -> float:
    base = dict(getattr(config, "RF_REGRESSION_PARAMS", {}))
    params = apply_site_rf_params(base, site)
    params["random_state"] = seed
    model = build_rf_regressor(params)
    model.fit(X_train, y_train)
    return float(model.predict(X_test)[0])


def _predict_xgb_rf_mode(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    seed: int,
) -> float:
    params = dict(XGB_RF_MODE_PARAMS)
    params["random_state"] = seed
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return float(model.predict(X_test)[0])


def _predict_stacking(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    site: str,
    seed: int,
) -> float:
    """
    Two-level stacking with chronological OOF (TimeSeriesSplit).

    Level 0: XGBoost + RF trained with TimeSeriesSplit(n_splits=5) to
             produce OOF predictions.
    Level 1: Ridge meta-learner trained on [xgb_oof, rf_oof] targets.
    Test:    Both base models retrained on full training set; predictions
             stacked and passed to the fitted Ridge meta-learner.

    All OOF computation happens within the pre-anchor training window.
    No future data is ever seen during meta-learner training.
    """
    n = len(y_train)

    # Fall back to rf_standard if too few rows for OOF
    if n < STACKING_MIN_TRAINING:
        return _predict_rf_standard(X_train, y_train, X_test, site, seed)

    # --- Level 0: generate OOF predictions ---
    n_splits = min(STACKING_OOF_SPLITS, n - 1)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    xgb_oof = np.zeros(n, dtype=float)
    rf_oof = np.zeros(n, dtype=float)

    xgb_base_params = dict(config.XGB_REGRESSION_PARAMS)
    xgb_base_params["n_jobs"] = 1
    xgb_base_params["random_state"] = seed
    xgb_base_params = apply_site_xgb_params(xgb_base_params, site)
    xgb_base_params.setdefault("nthread", 1)

    rf_base_params = dict(getattr(config, "RF_REGRESSION_PARAMS", {}))
    rf_base_params = apply_site_rf_params(rf_base_params, site)
    rf_base_params["random_state"] = seed

    for train_idx, val_idx in tscv.split(X_train):
        X_tr = X_train[train_idx] if isinstance(X_train, np.ndarray) else X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx] if hasattr(y_train, "iloc") else y_train[train_idx]
        X_val = X_train[val_idx] if isinstance(X_train, np.ndarray) else X_train.iloc[val_idx]

        try:
            xgb_fold = build_xgb_regressor(xgb_base_params)
            xgb_fold.fit(X_tr, y_tr)
            xgb_oof[val_idx] = xgb_fold.predict(X_val)
        except Exception:
            xgb_oof[val_idx] = float(np.mean(y_tr))

        try:
            rf_fold = build_rf_regressor(rf_base_params)
            rf_fold.fit(X_tr, y_tr)
            rf_oof[val_idx] = rf_fold.predict(X_val)
        except Exception:
            rf_oof[val_idx] = float(np.mean(y_tr))

    # --- Level 1: train Ridge meta-learner on OOF ---
    meta_X_train = np.column_stack([xgb_oof, rf_oof])
    meta_y_train = y_train.values if hasattr(y_train, "values") else np.array(y_train)
    ridge = Ridge(alpha=1.0)
    ridge.fit(meta_X_train, meta_y_train)

    # --- Test time: retrain base models on full training set ---
    xgb_full = build_xgb_regressor(xgb_base_params)
    xgb_full.fit(X_train, y_train)
    xgb_test_pred = float(xgb_full.predict(X_test)[0])

    rf_full = build_rf_regressor(rf_base_params)
    rf_full.fit(X_train, y_train)
    rf_test_pred = float(rf_full.predict(X_test)[0])

    meta_X_test = np.array([[xgb_test_pred, rf_test_pred]])
    stacked_pred = float(ridge.predict(meta_X_test)[0])
    return stacked_pred


# ===========================================================================
# Core: single-point evaluation (runs all 4 variants at once for efficiency)
# ===========================================================================

def _evaluate_single_point(
    raw_measurement: dict,
    feature_frame: pd.DataFrame,
    seed: int,
    use_per_site: bool = True,
) -> Optional[dict]:
    """
    Evaluate all four model variants for a single test point.
    Returns a dict with predictions for each variant, or None on failure.
    """
    test_date = pd.Timestamp(raw_measurement["date"])
    site = raw_measurement["site"]
    actual_da = float(raw_measurement["da_raw"])
    anchor_date = test_date - pd.Timedelta(days=FORECAST_HORIZON_DAYS)

    # --- Training data (up to anchor_date) ---
    train_data = get_site_training_frame(
        feature_frame, site, anchor_date, MIN_TRAINING_SAMPLES
    )
    if train_data is None or train_data.empty:
        return None

    # --- Test row (env features from anchor; persistence from train only) ---
    test_row = get_site_anchor_row(
        feature_frame, site, test_date, anchor_date, max_date_diff_days=28
    )
    if test_row is None:
        return None

    # Recompute persistence features from real observations only
    if "_is_interpolated" in train_data.columns:
        real_train = train_data[~train_data["_is_interpolated"]]
    else:
        real_train = train_data

    test_row = recompute_test_row_persistence_features(
        test_row, real_train, SPIKE_THRESHOLD
    )

    # --- Temporal features ---
    train_data = add_temporal_features(train_data)
    test_row = add_temporal_features(test_row)

    # --- Naive persistence (always real observations only) ---
    naive_val = get_last_known_raw_da(
        real_train,
        anchor_date=anchor_date,
        max_age_days=getattr(config, "PERSISTENCE_MAX_DAYS", None),
    )
    if naive_val is None:
        return None

    # --- Shared feature preparation ---
    try:
        X_train, y_train, X_test = _prepare_features(
            train_data, test_row, site, use_per_site
        )
    except Exception:
        return None

    # --- Leakage check ---
    anchor_ts = pd.Timestamp(anchor_date)
    if train_data["date"].max() > anchor_ts:
        return None
    if test_date <= anchor_ts:
        return None

    clip_q_global = getattr(config, "PREDICTION_CLIP_Q", 0.99)

    def _clip(v: float) -> float:
        return _postprocess(v, y_train, site, use_per_site, clip_q_global)

    results: dict = {
        "test_date": test_date,
        "anchor_date": anchor_date,
        "site": site,
        "actual_da": actual_da,
        "n_train": len(train_data),
    }

    # --- Run each variant with timing ---
    for variant in MODEL_NAMES:
        t0 = time.perf_counter()
        try:
            if variant == "xgb_standard":
                raw_pred = _predict_xgb_standard(X_train, y_train, X_test, site, seed)
            elif variant == "rf_standard":
                raw_pred = _predict_rf_standard(X_train, y_train, X_test, site, seed)
            elif variant == "xgb_rf_mode":
                raw_pred = _predict_xgb_rf_mode(X_train, y_train, X_test, seed)
            elif variant == "stacking":
                raw_pred = _predict_stacking(X_train, y_train, X_test, site, seed)
            else:
                continue
            pred = _clip(raw_pred)
        except Exception:
            pred = float(naive_val)   # degrade gracefully
        elapsed = time.perf_counter() - t0

        results[f"pred_{variant}"] = pred
        results[f"time_{variant}"] = elapsed

    return results


# ===========================================================================
# Module-level worker for multiprocessing (must not be nested)
# ===========================================================================

def _evaluate_anchor(args_tuple: tuple) -> Optional[dict]:
    """
    Worker function for ProcessPoolExecutor.  Receives everything it needs in a
    single tuple so it can be pickled cleanly by multiprocessing.

    args_tuple layout:
        (anchor_idx, raw_measurement, feature_frame_slice, base_seed, use_per_site)

    anchor_idx        – int; used to derive a deterministic per-anchor seed
    raw_measurement   – dict with keys 'date', 'site', 'da_raw'
    feature_frame_slice – DataFrame containing only the rows needed for this
                          anchor (site rows up to test_date + a small buffer);
                          keeps inter-process memory footprint small
    base_seed         – int; global seed from CLI
    use_per_site      – bool; mirrors the top-level config flag
    """
    # Ensure single-threaded models inside worker to prevent oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    anchor_idx, raw_measurement, feature_frame_slice, base_seed, use_per_site = args_tuple

    # Deterministic per-anchor seed for reproducibility
    rng = np.random.default_rng(base_seed + anchor_idx)
    anchor_seed = int(rng.integers(0, 2**31 - 1))

    return _evaluate_single_point(
        raw_measurement,
        feature_frame_slice,
        anchor_seed,
        use_per_site,
    )


# ===========================================================================
# Metric computation helpers
# ===========================================================================

def _compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    """Regression metrics + spike detection metrics."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    act = actual[mask]
    pred = predicted[mask]

    if len(act) < 2:
        return {
            "r2": float("nan"),
            "mae": float("nan"),
            "spike_recall": float("nan"),
            "spike_precision": float("nan"),
            "spike_f1": float("nan"),
            "transition_recall": float("nan"),
            "n": int(len(act)),
        }

    r2 = float(r2_score(act, pred))
    mae = float(mean_absolute_error(act, pred))

    # Spike detection at DA >= 20 threshold
    spike_actual = (act >= SPIKE_THRESHOLD).astype(int)
    spike_pred = (pred >= SPIKE_THRESHOLD).astype(int)

    if spike_actual.sum() == 0:
        spike_recall = float("nan")
        spike_precision = float("nan")
        spike_f1 = float("nan")
    else:
        spike_recall = float(recall_score(spike_actual, spike_pred, zero_division=0))
        spike_precision = float(precision_score(spike_actual, spike_pred, zero_division=0))
        spike_f1 = float(f1_score(spike_actual, spike_pred, zero_division=0))

    # Transition recall: below-to-above crossings of 20 µg/g
    # A "transition" is a consecutive pair where actual[i-1] < 20 and actual[i] >= 20
    transitions = 0
    transitions_caught = 0
    for i in range(1, len(act)):
        if act[i - 1] < SPIKE_THRESHOLD and act[i] >= SPIKE_THRESHOLD:
            transitions += 1
            if pred[i] >= SPIKE_THRESHOLD:
                transitions_caught += 1

    transition_recall = (
        float(transitions_caught / transitions) if transitions > 0 else float("nan")
    )

    return {
        "r2": r2,
        "mae": mae,
        "spike_recall": spike_recall,
        "spike_precision": spike_precision,
        "spike_f1": spike_f1,
        "transition_recall": transition_recall,
        "n": int(len(act)),
    }


# ===========================================================================
# Test point sampling (mirrors raw_forecast_engine's sampling logic)
# ===========================================================================

def _sample_test_points(
    raw_data: pd.DataFrame,
    feature_frame: pd.DataFrame,
    sample_fraction: float,
    seed: int,
    site_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Sample test points per site, applying:
      - MIN_TEST_DATE filter
      - History requirement: >= 33% of site total measurements must precede anchor
      - sample_fraction of valid candidates per site
    """
    min_test_ts = pd.Timestamp(getattr(config, "MIN_TEST_DATE", "2003-01-01"))
    history_frac = getattr(config, "HISTORY_REQUIREMENT_FRACTION", 0.33)

    candidate_raw = raw_data[raw_data["date"] >= min_test_ts].copy()

    if site_filter:
        candidate_raw = candidate_raw[candidate_raw["site"].isin(site_filter)]

    site_total_counts = raw_data.groupby("site")["date"].size().to_dict()

    valid_rows = []
    for _, row in candidate_raw.iterrows():
        anchor_dt = row["date"] - pd.Timedelta(days=FORECAST_HORIZON_DAYS)
        site = row["site"]
        total_site = site_total_counts.get(site, 0)
        if total_site == 0:
            continue
        min_required = max(int(np.ceil(history_frac * total_site)), MIN_TRAINING_SAMPLES)
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
        if len(site_history) >= MIN_TRAINING_SAMPLES:
            valid_rows.append(row)

    if not valid_rows:
        return pd.DataFrame()

    valid_df = pd.DataFrame(valid_rows)

    rng = np.random.RandomState(seed)
    sampled_rows = []
    for site, site_df in valid_df.groupby("site"):
        site_df = site_df.sort_values("date")
        n_candidates = len(site_df)
        total_site = site_total_counts.get(site, n_candidates)
        target = min(int(np.ceil(sample_fraction * total_site)), n_candidates)
        target = max(target, 1)
        idx = rng.choice(n_candidates, size=min(target, n_candidates), replace=False)
        sampled_rows.append(site_df.iloc[idx])

    if not sampled_rows:
        return pd.DataFrame()

    return pd.concat(sampled_rows, ignore_index=True)


# ===========================================================================
# Main evaluation routine
# ===========================================================================

def run_evaluation(
    sample_fraction: float = 0.40,
    seed: int = 42,
    sites_arg: str = "all",
    output_dir: str = "eval_results/model_variants",
    n_workers: Optional[int] = None,
) -> None:
    """Run the full evaluation and write outputs."""
    os.makedirs(output_dir, exist_ok=True)

    max_workers = n_workers if n_workers is not None else min(os.cpu_count() or 4, 8)

    print(f"\n{'='*70}")
    print("DATect Model Variant Comparison")
    print(f"  Variants  : {', '.join(MODEL_NAMES)}")
    print(f"  Fraction  : {sample_fraction:.0%} of eligible test points per site")
    print(f"  Seed      : {seed}")
    print(f"  Sites     : {sites_arg}")
    print(f"  Workers   : {max_workers}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*70}\n")

    # --- Load data ---
    print("Loading raw DA measurements...")
    raw_data = load_raw_da_measurements()

    print("Loading processed feature parquet...")
    processed = pd.read_parquet(config.FINAL_OUTPUT_PATH)
    processed["date"] = pd.to_datetime(processed["date"])
    processed = processed.sort_values(["site", "date"]).reset_index(drop=True)

    from forecasting.feature_utils import add_temporal_features as _atf
    processed = _atf(processed)

    raw_weekly = aggregate_raw_to_weekly(raw_data)
    feature_frame = build_raw_feature_frame(processed, raw_weekly)

    # --- Resolve site filter ---
    WA_SITES = {"Kalaloch", "Quinault", "Copalis", "Twin Harbors", "Long Beach"}
    all_sites = list(config.SITES.keys())

    if sites_arg.lower() == "all":
        site_filter = None  # no filter
    elif sites_arg.upper() == "WA":
        site_filter = [s for s in all_sites if s in WA_SITES]
    else:
        # Treat as comma-separated or single site name
        requested = [s.strip() for s in sites_arg.split(",")]
        site_filter = []
        for req in requested:
            matches = [s for s in all_sites if s.lower() == req.lower()]
            if matches:
                site_filter.extend(matches)
            else:
                print(f"  WARNING: site '{req}' not found in config.SITES; skipping.")
        if not site_filter:
            print("ERROR: no matching sites found. Exiting.")
            sys.exit(1)

    # --- Sample test points ---
    print("Sampling test points...")
    test_samples = _sample_test_points(
        raw_data, feature_frame, sample_fraction, seed, site_filter
    )
    if test_samples.empty:
        print("ERROR: No valid test samples found. Check data files and site names.")
        sys.exit(1)
    print(f"  {len(test_samples)} test points sampled across "
          f"{test_samples['site'].nunique()} site(s).\n")

    sample_rows = [
        {"date": row["date"], "site": row["site"], "da_raw": row["da_raw"]}
        for _, row in test_samples.iterrows()
    ]

    # --- Build per-anchor args (pre-slice feature_frame to reduce IPC payload) ---
    use_per_site = getattr(config, "USE_PER_SITE_MODELS", True)

    anchor_args_list: List[tuple] = []
    for anchor_idx, raw_meas in enumerate(sample_rows):
        site = raw_meas["site"]
        test_date = pd.Timestamp(raw_meas["date"])
        # Include a small buffer (28 days) past test_date so anchor-row lookup works
        slice_end = test_date + pd.Timedelta(days=28)
        ff_slice = feature_frame[
            (feature_frame["site"] == site) & (feature_frame["date"] <= slice_end)
        ].copy()
        anchor_args_list.append((anchor_idx, raw_meas, ff_slice, seed, use_per_site))

    # --- Run evaluations (parallel with ProcessPoolExecutor, sequential fallback) ---
    all_results: List[dict] = []

    try:
        import multiprocessing
        mp_context = multiprocessing.get_context("spawn")
        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
            for args in anchor_args_list:
                future = executor.submit(_evaluate_anchor, args)
                futures[future] = args[0]  # anchor_idx as key

            with tqdm(total=len(futures), desc="Evaluating", unit="point") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as exc:
                        warnings.warn(
                            f"Anchor {futures[future]} raised an exception: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        result = None
                    if result is not None:
                        all_results.append(result)
                    pbar.update(1)

    except Exception as parallel_exc:
        warnings.warn(
            f"ProcessPoolExecutor failed ({parallel_exc}); "
            "falling back to sequential evaluation.",
            RuntimeWarning,
            stacklevel=2,
        )
        all_results = []
        for args in tqdm(anchor_args_list, desc="Evaluating (sequential)", unit="point"):
            result = _evaluate_anchor(args)
            if result is not None:
                all_results.append(result)

    if not all_results:
        print("ERROR: No predictions were produced. Check data and configuration.")
        sys.exit(1)

    results_df = pd.DataFrame(all_results)
    n_total = len(results_df)
    print(f"\n  {n_total} successful predictions (out of {len(sample_rows)} attempted).\n")

    # --- Compute metrics ---
    actual = results_df["actual_da"].values

    # Overall metrics
    overall_metrics: Dict[str, dict] = {}
    timing_stats: Dict[str, dict] = {}

    for variant in MODEL_NAMES:
        pred_col = f"pred_{variant}"
        time_col = f"time_{variant}"
        if pred_col not in results_df.columns:
            continue
        predicted = results_df[pred_col].values
        overall_metrics[variant] = _compute_metrics(actual, predicted)

        times = results_df[time_col].values
        timing_stats[variant] = {
            "mean_s": float(np.nanmean(times)),
            "median_s": float(np.nanmedian(times)),
            "total_s": float(np.nansum(times)),
        }

    # Per-site metrics
    site_metrics: Dict[str, Dict[str, dict]] = defaultdict(dict)
    per_site_rows = []

    for site, site_df in results_df.groupby("site"):
        site_actual = site_df["actual_da"].values
        for variant in MODEL_NAMES:
            pred_col = f"pred_{variant}"
            if pred_col not in site_df.columns:
                continue
            site_pred = site_df[pred_col].values
            m = _compute_metrics(site_actual, site_pred)
            site_metrics[site][variant] = m
            per_site_rows.append({
                "site": site,
                "variant": variant,
                "n": m["n"],
                "r2": m["r2"],
                "mae": m["mae"],
                "spike_recall": m["spike_recall"],
                "spike_precision": m["spike_precision"],
                "spike_f1": m["spike_f1"],
                "transition_recall": m["transition_recall"],
                "mean_train_time_s": timing_stats[variant]["mean_s"],
            })

    # --- Print comparison table ---
    _print_comparison_table(overall_metrics, timing_stats, site_metrics)

    # --- Write outputs ---
    variant_csv_path = os.path.join(output_dir, "variant_comparison.csv")
    per_site_df = pd.DataFrame(per_site_rows)
    per_site_df.to_csv(variant_csv_path, index=False)
    print(f"\nPer-site CSV written to: {variant_csv_path}")

    summary = {
        "overall": overall_metrics,
        "timing": timing_stats,
        "n_test_points": n_total,
        "sample_fraction": sample_fraction,
        "seed": seed,
        "sites": sites_arg,
        "note_stacking": (
            "Stacking uses 5-fold chronological TimeSeriesSplit OOF entirely "
            "within the pre-anchor training window. No future data is used. "
            "Expect ~2-3x slower than single models."
        ),
    }
    summary_json_path = os.path.join(output_dir, "variant_summary.json")
    with open(summary_json_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    print(f"Summary JSON written to:  {summary_json_path}")


# ===========================================================================
# Output formatting
# ===========================================================================

def _fmt(v, fmt=".4f") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "   N/A  "
    return format(v, fmt)


def _print_comparison_table(
    overall: Dict[str, dict],
    timing: Dict[str, dict],
    site_metrics: Dict[str, Dict[str, dict]],
) -> None:
    col_w = 14
    metric_keys = ["r2", "mae", "spike_recall", "spike_precision", "spike_f1", "transition_recall"]
    metric_labels = ["R²", "MAE", "Spike Recall", "Spike Prec.", "Spike F1", "Trans. Recall"]

    header = f"{'Metric':<20}" + "".join(f"{n:>{col_w}}" for n in MODEL_NAMES)
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("OVERALL METRICS")
    print("=" * len(header))
    print(header)
    print(sep)

    for key, label in zip(metric_keys, metric_labels):
        row = f"{label:<20}"
        for variant in MODEL_NAMES:
            val = overall.get(variant, {}).get(key, float("nan"))
            row += f"{_fmt(val):>{col_w}}"
        print(row)

    print(sep)
    print(f"{'N (test pts)':<20}" + "".join(
        f"{overall.get(v, {}).get('n', 0):>{col_w}}" for v in MODEL_NAMES
    ))

    print("\n" + "-" * len(header))
    print("TRAINING TIME (seconds per anchor point)")
    print("-" * len(header))
    for label, key in [("Mean", "mean_s"), ("Median", "median_s"), ("Total", "total_s")]:
        row = f"{label:<20}"
        for variant in MODEL_NAMES:
            val = timing.get(variant, {}).get(key, float("nan"))
            row += f"{_fmt(val, '.3f'):>{col_w}}"
        print(row)

    # Per-site table
    all_sites = sorted(site_metrics.keys())
    if not all_sites:
        return

    print("\n" + "=" * len(header))
    print("PER-SITE R²")
    print("=" * len(header))
    site_header = f"{'Site':<22}" + "".join(f"{n:>{col_w}}" for n in MODEL_NAMES)
    print(site_header)
    print("-" * len(site_header))
    for site in all_sites:
        row = f"{site:<22}"
        for variant in MODEL_NAMES:
            val = site_metrics[site].get(variant, {}).get("r2", float("nan"))
            row += f"{_fmt(val):>{col_w}}"
        print(row)

    print("\n" + "=" * len(header))
    print("PER-SITE MAE (µg/g)")
    print("=" * len(header))
    print(site_header)
    print("-" * len(site_header))
    for site in all_sites:
        row = f"{site:<22}"
        for variant in MODEL_NAMES:
            val = site_metrics[site].get(variant, {}).get("mae", float("nan"))
            row += f"{_fmt(val):>{col_w}}"
        print(row)

    print("\n" + "=" * len(header))
    print("PER-SITE SPIKE F1 (DA ≥ 20 µg/g)")
    print("=" * len(header))
    print(site_header)
    print("-" * len(site_header))
    for site in all_sites:
        row = f"{site:<22}"
        for variant in MODEL_NAMES:
            val = site_metrics[site].get(variant, {}).get("spike_f1", float("nan"))
            row += f"{_fmt(val):>{col_w}}"
        print(row)

    print()


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare xgb_standard, rf_standard, xgb_rf_mode, and stacking "
                    "under DATect's leak-free anchor-based retrospective protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling and model training.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.40,
        help=(
            "Fraction of eligible raw measurements per site to use as test points. "
            "Eligibility requires >= 33%% of site history to precede the anchor date."
        ),
    )
    parser.add_argument(
        "--sites",
        type=str,
        default="all",
        help=(
            "Which sites to include. "
            "'all' = all 10 Pacific Coast sites, "
            "'WA' = Washington state sites only, "
            "or a comma-separated list of site names (e.g. 'Newport,Coos Bay')."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results/model_variants",
        help="Directory for output files (CSV and JSON).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel worker processes for the anchor evaluation loop. "
            "Defaults to min(cpu_count, 8). Set to 1 to force sequential execution."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        sample_fraction=args.sample_fraction,
        seed=args.seed,
        sites_arg=args.sites,
        output_dir=args.output_dir,
        n_workers=args.workers,
    )

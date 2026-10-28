#!/usr/bin/env python3
"""
Tiny causal sequence baseline: sklearn MLP on flattened last-K weekly feature windows.

Reuses the same leak-free sample selection and row builders as
:class:`forecasting.raw_forecast_engine.RawForecastEngine` (training data
``date <= anchor``, persistence recomputed from real DA only, same
``create_transformer`` path). Compare pooled R²/MAE on raw test DA to
``quick_raw_retrospective_compare.py`` (naive / Ridge / ensemble).

Does not use dense panel imputation metrics.

Usage (repo root):
    python3 scripts/eval/tiny_sequence_baseline.py --max-samples 40 --window 6
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _select_retrospective_samples(
    feature_frame: pd.DataFrame,
    raw_data: pd.DataFrame,
    *,
    random_seed: int,
    min_test_date: str,
    min_training: int,
    max_test_samples: Optional[int],
) -> pd.DataFrame:
    """Mirror RawForecastEngine retrospective candidate selection (no model calls)."""
    import config

    min_test_ts = pd.Timestamp(getattr(config, "MIN_TEST_DATE", min_test_date))
    forecast_horizon = config.FORECAST_HORIZON_DAYS
    history_frac = getattr(config, "HISTORY_REQUIREMENT_FRACTION", 0.33)

    candidate_raw = raw_data[raw_data["date"] >= min_test_ts].copy()
    site_total_counts = raw_data.groupby("site")["date"].size().to_dict()

    valid_rows = []
    for _, row in candidate_raw.iterrows():
        anchor_dt = row["date"] - pd.Timedelta(days=forecast_horizon)
        site = row["site"]
        total_site = site_total_counts.get(site, 0)
        if total_site == 0:
            continue
        min_required = max(
            int(np.ceil(history_frac * total_site)),
            min_training,
        )
        n_history = len(
            raw_data[
                (raw_data["site"] == site) & (raw_data["date"] <= anchor_dt)
            ]
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
        return pd.DataFrame()

    valid_df = pd.DataFrame(valid_rows)
    rng = np.random.RandomState(random_seed)
    sampled_rows = []
    for site, site_df in valid_df.groupby("site"):
        site_df = site_df.sort_values("date")
        n_candidates = len(site_df)
        total_site = site_total_counts.get(site, n_candidates)
        test_frac = getattr(config, "TEST_SAMPLE_FRACTION", 0.20)
        target = min(int(np.ceil(test_frac * total_site)), n_candidates)
        if target <= 0:
            continue
        idx = rng.choice(n_candidates, size=target, replace=False)
        sampled_rows.append(site_df.iloc[idx])

    if not sampled_rows:
        return pd.DataFrame()

    test_samples = pd.concat(sampled_rows, ignore_index=True)
    if max_test_samples is not None and len(test_samples) > max_test_samples:
        test_samples = test_samples.sample(
            n=max_test_samples,
            random_state=random_seed,
        ).reset_index(drop=True)
    return test_samples


def _postprocess(
    value: float,
    train_data: pd.DataFrame,
    site: str,
    *,
    use_per_site: bool,
) -> float:
    import config
    from forecasting.per_site_models import get_site_clip_params

    value = max(0.0, value)
    clip_q = getattr(config, "PREDICTION_CLIP_Q", 0.99)
    if use_per_site:
        sq, sm = get_site_clip_params(site)
        cq = sq if sq is not None else clip_q
    else:
        cq = clip_q
        sm = None
    if cq is not None:
        clip_max = float(np.quantile(train_data["da_raw"], cq))
        value = min(value, clip_max)
    if sm is not None:
        value = min(value, sm)
    return float(value)


def _mlp_predict_one(
    raw_measurement: dict,
    feature_frame: pd.DataFrame,
    *,
    min_training: int,
    window: int,
    mlp_seed: int,
) -> Optional[float]:
    import config
    from forecasting.feature_utils import add_temporal_features, create_transformer
    from forecasting.per_site_models import compute_site_drop_cols
    from forecasting.raw_data_forecaster import (
        get_site_anchor_row,
        get_site_training_frame,
        recompute_test_row_persistence_features,
    )
    from forecasting.raw_forecast_engine import _verify_no_data_leakage

    test_date = raw_measurement["date"]
    site = raw_measurement["site"]
    forecast_horizon = config.FORECAST_HORIZON_DAYS
    anchor_date = test_date - pd.Timedelta(days=forecast_horizon)
    use_per_site = getattr(config, "USE_PER_SITE_MODELS", True)
    zero_imp = getattr(config, "ZERO_IMPORTANCE_FEATURES", [])

    train_data = get_site_training_frame(
        feature_frame, site, anchor_date, min_training
    )
    if train_data is None or train_data.empty:
        return None

    test_row = get_site_anchor_row(
        feature_frame, site, test_date, anchor_date, max_date_diff_days=28
    )
    if test_row is None:
        return None

    if "_is_interpolated" in train_data.columns:
        real_train = train_data[~train_data["_is_interpolated"]]
    else:
        real_train = train_data
    test_row = recompute_test_row_persistence_features(
        test_row, real_train, config.SPIKE_THRESHOLD
    )

    train_data = add_temporal_features(train_data)
    test_row = add_temporal_features(test_row)

    drop_cols = ["date", "site", "da_raw", "da", "_is_interpolated"] + list(zero_imp)
    if use_per_site:
        drop_cols = compute_site_drop_cols(
            drop_cols, train_data.columns.tolist(), site
        )

    try:
        transformer, X_train = create_transformer(train_data, drop_cols)
        y_train = train_data["da_raw"].astype(float).copy()
        X_train_p = transformer.fit_transform(X_train)
        _verify_no_data_leakage(train_data, test_date, anchor_date)
    except Exception:
        return None

    n = len(X_train_p)
    if n < window:
        return None

    arr = np.asarray(X_train_p)
    X_seq = []
    y_seq = []
    for j in range(window - 1, n):
        X_seq.append(arr[j - window + 1 : j + 1].ravel())
        y_seq.append(float(y_train.iloc[j]))
    if len(X_seq) < min_training:
        return None

    mlp = MLPRegressor(
        hidden_layer_sizes=(48,),
        max_iter=500,
        early_stopping=True,
        random_state=mlp_seed,
        alpha=1e-2,
    )
    try:
        mlp.fit(X_seq, y_seq)
    except Exception:
        return None

    test_window = arr[n - window : n].ravel().reshape(1, -1)
    try:
        raw_pred = float(mlp.predict(test_window)[0])
    except Exception:
        return None

    return _postprocess(raw_pred, train_data, site, use_per_site=use_per_site)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-test-date", type=str, default="2008-01-01")
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--window", type=int, default=6, help="Causal weeks in flattened input")
    args = parser.parse_args()

    import config as cfg

    orig = {
        "seed": cfg.RANDOM_SEED,
        "frac": getattr(cfg, "TEST_SAMPLE_FRACTION", 0.20),
        "parallel": getattr(cfg, "ENABLE_PARALLEL", True),
        "min_date": getattr(cfg, "MIN_TEST_DATE", "2003-01-01"),
    }
    cfg.RANDOM_SEED = args.seed
    cfg.TEST_SAMPLE_FRACTION = args.fraction
    cfg.MIN_TEST_DATE = args.min_test_date
    cfg.ENABLE_PARALLEL = False

    try:
        from forecasting.raw_forecast_engine import RawForecastEngine

        engine = RawForecastEngine(validate_on_init=False)
        engine.random_seed = args.seed
        feature_frame = engine._load_feature_frame()
        raw_data = engine._raw_data_cache
        min_tr = engine.min_training_samples

        test_samples = _select_retrospective_samples(
            feature_frame,
            raw_data,
            random_seed=args.seed,
            min_test_date=args.min_test_date,
            min_training=min_tr,
            max_test_samples=args.max_samples if args.max_samples > 0 else None,
        )
    finally:
        cfg.RANDOM_SEED = orig["seed"]
        cfg.TEST_SAMPLE_FRACTION = orig["frac"]
        cfg.ENABLE_PARALLEL = orig["parallel"]
        cfg.MIN_TEST_DATE = orig["min_date"]

    if test_samples.empty:
        print("ERROR: No test samples selected.")
        return 1

    preds = []
    actuals = []
    sites = []
    for _, row in test_samples.iterrows():
        meas = {"date": row["date"], "site": row["site"], "da_raw": row["da_raw"]}
        p = _mlp_predict_one(
            meas,
            feature_frame,
            min_training=min_tr,
            window=args.window,
            mlp_seed=args.seed,
        )
        if p is not None:
            preds.append(p)
            actuals.append(float(row["da_raw"]))
            sites.append(row["site"])

    if len(preds) < 2:
        print("ERROR: Too few successful MLP predictions.")
        return 1

    y_true = np.array(actuals)
    y_pred = np.array(preds)
    pooled = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": len(y_pred),
    }

    print("\n=== Tiny sequence baseline (MLP, causal K-week windows) ===")
    print(
        f"window={args.window}, fraction={args.fraction}, max_samples={args.max_samples}, "
        f"seed={args.seed}, successful_preds={pooled['n']}"
    )
    print(f"Pooled R2={pooled['r2']:.4f}  MAE={pooled['mae']:.3f}")
    print("\nPer-site R2:")
    site_arr = np.array(sites)
    for site in sorted(set(sites)):
        m = site_arr == site
        if m.sum() < 2:
            continue
        print(
            f"  {site:<16} R2={r2_score(y_true[m], y_pred[m]):.3f}  "
            f"n={int(m.sum())}"
        )
    print(
        "\nCompare to naive / linear / ensemble on the same draw using:\n"
        "  python3 scripts/eval/quick_raw_retrospective_compare.py "
        f"--fraction {args.fraction} --seed {args.seed} "
        f"--min-test-date {args.min_test_date} --max-samples {args.max_samples}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

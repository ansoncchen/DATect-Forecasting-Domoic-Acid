#!/usr/bin/env python3
"""Experiment harness for the one-week-ahead DA forecast (docs/FORECAST_EXPERIMENTS.md).

Re-implements ``RawForecastEngine._run_single_raw_validation`` with switches for
one change at a time, on the exact test points of the committed deterministic
chronological runs. With the ``baseline`` variant it reproduces the committed
ensemble predictions (checked by ``--verify``).

Unlike the engine it stores the *unclipped* XGBoost / Random Forest / Ridge
predictions plus several quantiles of the training targets, so clip, blend,
bias and threshold variants are pure rescoring (``rescore.py``). The spike
classifier is not refitted; the committed ``spike_probability`` is reused.

    python scripts/experiments/harness.py --window val --variants baseline,real_only
    python scripts/experiments/harness.py --window val --variants baseline --limit 40 --verify
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import config  # noqa: E402
from forecasting.feature_utils import add_temporal_features, create_transformer  # noqa: E402
from forecasting.per_site_models import (  # noqa: E402
    apply_site_rf_params, apply_site_xgb_params, compute_site_drop_cols,
    get_site_clip_params, get_site_ensemble_weights,
)
from forecasting.raw_data_forecaster import (  # noqa: E402
    get_last_known_raw_da, get_site_anchor_row, get_site_training_frame,
    recompute_test_row_persistence_features, week_complete_by,
)
from forecasting.raw_model_factory import (  # noqa: E402
    build_linear_regressor, build_rf_regressor, build_xgb_regressor,
)

WINDOWS = {
    "val": ("2019-01-01", "2022-01-01",
            "results/chronological/chronological_regression_ensemble_20190101_20220101.parquet"),
    "holdout": ("2022-01-01", "2024-01-01",
                "results/chronological/chronological_regression_ensemble_20220101_20240101.parquet"),
    # Secondary selection window: every real measurement 2015-2018 that passes the
    # engine's history filter (per-site configs were curated on pre-2019 data, so
    # this window is not independent of them; used only as a consistency check).
    "dev": ("2015-01-01", "2019-01-01", None),
}
QUANTILES = (0.90, 0.95, 0.97, 0.99, 1.0)
PERSIST_COLS = [
    "last_observed_da_raw", "weeks_since_last_spike",
    "da_raw_prev_obs_1", "da_raw_prev_obs_2", "da_raw_prev_obs_3", "da_raw_prev_obs_4",
    "da_raw_prev_obs_2_weeks_ago", "da_raw_prev_obs_3_weeks_ago", "da_raw_prev_obs_diff_1_2",
    "raw_obs_roll_mean_4", "raw_obs_roll_std_4", "raw_obs_roll_max_4",
    "raw_obs_roll_std_8", "raw_obs_roll_max_8", "raw_obs_roll_std_12", "raw_obs_roll_max_12",
]
PN_COLS = ["pn", "pn_log"]
OAD_PREFIX = "oad_"
ENV_COLS = ["oni", "pdo", "discharge", "beuti", "modis-flr", "modis-sst", "sst-anom",
            "chla-anom", "modis-chla", "modis-k490", "modis-par"]

# Each variant changes one thing relative to "baseline" (the committed engine).
VARIANTS: dict[str, dict] = {
    "baseline": {},
    # training rows / features
    "real_only": {"train_rows": "real"},
    "refresh": {"features": "refresh"},
    "asof": {"features": "asof"},
    "asof_gapfill": {"features": "asof", "asof_gapfill": True},
    "obs_age": {"add_obs_age": True},
    "no_pn": {"drop": PN_COLS},
    "no_oad": {"drop_prefix": OAD_PREFIX},
    "no_env": {"drop": PN_COLS + ENV_COLS, "drop_prefix": OAD_PREFIX},
    "no_lags": {"drop": [c for c in PERSIST_COLS if c.startswith("da_raw_prev_obs")]},
    "no_oni": {"drop": ["oni"]},
    "no_discharge": {"drop": ["discharge"]},
    "no_sat": {"drop": ["modis-sst", "sst-anom", "modis-flr"]},
    "all_features": {"feature_subset_mode": "all"},
    "no_monotone": {"monotone": False},
    "no_per_site": {"per_site": False},
    # targets / losses
    "log1p": {"target": "log1p"},
    "log1p_smear": {"target": "log1p", "smear": True},
    "sqrt": {"target": "sqrt"},
    "tweedie": {"xgb_extra": {"objective": "reg:tweedie", "tweedie_variance_power": 1.5},
                "rf_extra": {"criterion": "poisson"}},
    "poisson": {"xgb_extra": {"objective": "count:poisson"}, "rf_extra": {"criterion": "poisson"}},
    "huber10": {"xgb_extra": {"objective": "reg:pseudohubererror", "huber_slope": 10.0}},
    # sample weighting / window
    "recency_hl5": {"halflife_years": 5.0},
    "recency_hl10": {"halflife_years": 10.0},
    "window_8y": {"train_years": 8.0},
    "spike_weight3": {"spike_weight": 3.0},
    # pooled cross-site model (XGBoost only, site one-hot, all features)
    "pooled": {"pooled": True},
    # hyperparameters
    "xgb_shallow": {"xgb_extra": {"max_depth": 3, "min_child_weight": 5}},
    "xgb_more_trees": {"xgb_extra_scale_trees": 2.0},
    "rf_leaf1": {"rf_extra": {"min_samples_leaf": 1, "min_samples_split": 2}},
    # noise floor: identical configuration, different model seed
    "seed43": {"seed": 43},
    "seed44": {"seed": 44},
    # pooled spike classifier only (combined with a regression run in rescore.py)
    "cls_baseline": {"classifier_only": True},
    "cls_safe_fixed": {"classifier_only": True, "cls_safe_fixed": True},
    "cls_refresh": {"classifier_only": True, "cls_features": "refresh"},
    "cls_asof": {"classifier_only": True, "cls_features": "asof"},
    "cls_asof_safe_fixed": {"classifier_only": True, "cls_features": "asof", "cls_safe_fixed": True},
}


# ---------------------------------------------------------------------------
# Context (built once per process; joblib workers build their own)
# ---------------------------------------------------------------------------

_CTX: dict | None = None


def context() -> dict:
    global _CTX
    if _CTX is None:
        from forecasting.raw_forecast_engine import RawForecastEngine
        eng = RawForecastEngine(validate_on_init=False)
        ff = eng._load_feature_frame()
        raw = eng._raw_data_cache.copy()
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.sort_values(["site", "date"]).reset_index(drop=True)
        obs = {s: (g["date"].to_numpy(), g["da_raw"].to_numpy(float)) for s, g in raw.groupby("site")}
        _CTX = {"ff": ff, "raw": raw, "obs": obs, "asof": {}}
    return _CTX


def monday(ts: pd.Timestamp) -> pd.Timestamp:
    return ts - pd.Timedelta(days=ts.weekday())


def asof_persistence(od: np.ndarray, ov: np.ndarray, cutoff, ref, grid_monday) -> dict:
    """Persistence features from observations dated <= ``cutoff`` (inclusive).

    Same definitions as build_raw_feature_frame / create_raw_lag_features, but for an
    arbitrary information cutoff: lags are the most recent observations, "weeks ago"
    and weeks_since_last_spike are measured from ``ref``, and the rolling statistics
    run over the weekly grid ending at ``grid_monday`` (value at week X = the last
    observation dated before X and on or before the cutoff).
    """
    cutoff = np.datetime64(pd.Timestamp(cutoff))
    ref = pd.Timestamp(ref)
    idx = int(np.searchsorted(od, cutoff, side="right"))
    f = {c: np.nan for c in PERSIST_COLS}
    f["weeks_since_last_spike"] = 999.0
    if idx == 0:
        return f
    vals, dts = ov[:idx], od[:idx]
    f["last_observed_da_raw"] = vals[-1]
    spikes = np.flatnonzero(vals >= config.SPIKE_THRESHOLD)
    if len(spikes):
        f["weeks_since_last_spike"] = (ref - pd.Timestamp(dts[spikes[-1]])).days / 7.0
    for i in range(1, 5):
        if idx >= i:
            f[f"da_raw_prev_obs_{i}"] = vals[-i]
            if i in (2, 3):
                f[f"da_raw_prev_obs_{i}_weeks_ago"] = (ref - pd.Timestamp(dts[-i])).days / 7.0
    f["da_raw_prev_obs_diff_1_2"] = f["da_raw_prev_obs_1"] - f["da_raw_prev_obs_2"]
    gm = pd.Timestamp(grid_monday)
    for k in (4, 8, 12):
        grid = np.array([np.datetime64(gm - pd.Timedelta(days=7 * j)) for j in range(k)])
        j_idx = np.minimum(np.searchsorted(od, grid, side="left"), idx)
        v = np.where(j_idx > 0, ov[np.maximum(j_idx - 1, 0)], np.nan)
        v = v[~np.isnan(v)]
        if len(v):
            if k == 4:
                f["raw_obs_roll_mean_4"] = v.mean()
            f[f"raw_obs_roll_max_{k}"] = v.max()
            f[f"raw_obs_roll_std_{k}"] = v.std(ddof=1) if len(v) > 1 else np.nan
    return f


def weekly_obs_upto(od: np.ndarray, ov: np.ndarray, cutoff) -> tuple[np.ndarray, np.ndarray]:
    """Weekly-max observations (dated by their Monday) using only days <= cutoff."""
    m = od <= np.datetime64(pd.Timestamp(cutoff))
    if not m.any():
        return od[:0], ov[:0]
    s = pd.Series(ov[m], index=pd.DatetimeIndex(od[m]))
    wk = s.groupby(s.index - pd.to_timedelta(s.index.weekday, unit="D")).max()
    return wk.index.to_numpy(), wk.to_numpy(float)


def env_row(ff_site: pd.DataFrame, anchor) -> pd.Series | None:
    """Latest weekly row whose week ends by ``anchor`` (as get_site_anchor_row)."""
    past = ff_site[week_complete_by(ff_site["date"], anchor)]
    if past.empty:
        return None
    row = past.iloc[-1]
    if abs((row["date"] - pd.Timestamp(anchor)).days) > 28:
        return None
    return row


def asof_table(site: str, gapfill: bool) -> pd.DataFrame:
    """Training rows built exactly like a forecast row: features as of (target - 7 days).

    One row per real measurement (site-day, daily max). With ``gapfill`` the weekly
    gap-filled rows are added with a nominal target day of Thursday.
    """
    ctx = context()
    key = (site, gapfill)
    if key in ctx["asof"]:
        return ctx["asof"][key]
    ff_site = ctx["ff"][ctx["ff"]["site"] == site].sort_values("date")
    od, ov = ctx["obs"][site]
    rows = []
    targets = [(pd.Timestamp(d), float(v), False) for d, v in zip(od, ov)]
    if gapfill:
        gf = ff_site[ff_site["da_raw"].isna() & ff_site["da"].notna()]
        targets += [(t + pd.Timedelta(days=3), float(v), True) for t, v in zip(gf["date"], gf["da"])]
    for d, y, interp in targets:
        a = d - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
        er = env_row(ff_site, a)
        if er is None:
            continue
        r = er.to_dict()
        r.update(asof_persistence(od, ov, a, d, monday(d)))
        r.update({"date": d, "da_raw": y, "_is_interpolated": interp,
                  "_known_by": d if not interp else monday(d) + pd.Timedelta(days=6)})
        rows.append(r)
    tab = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    ctx["asof"][key] = tab
    return tab


# ---------------------------------------------------------------------------
# One forecast
# ---------------------------------------------------------------------------

def _xgb_params(site: str, v: dict) -> dict:
    params = dict(config.XGB_REGRESSION_PARAMS)
    params["n_jobs"] = 1
    if v.get("per_site", True):
        params = apply_site_xgb_params(params, site)
    params.update(v.get("xgb_extra", {}))
    if "xgb_extra_scale_trees" in v:
        params["n_estimators"] = int(params["n_estimators"] * v["xgb_extra_scale_trees"])
    return params


def _rf_params(site: str, v: dict) -> dict:
    params = dict(config.RF_REGRESSION_PARAMS)
    if v.get("per_site", True):
        params = apply_site_rf_params(params, site)
    params.update(v.get("rf_extra", {}))
    return params


def _weights(site: str, v: dict) -> tuple[float, float, float]:
    if not v.get("per_site", True):
        return 0.5, 0.5, 0.0
    return get_site_ensemble_weights(site)


def _drop_cols(train: pd.DataFrame, site: str, v: dict) -> list[str]:
    drop = ["date", "site", "da_raw", "da", "_is_interpolated", "_known_by"] + list(config.ZERO_IMPORTANCE_FEATURES)
    if v.get("per_site", True) and v.get("feature_subset_mode") != "all":
        cols = train.columns.tolist()
        extra = ["days_since_last_obs"] if v.get("add_obs_age") else []
        drop = compute_site_drop_cols(drop, cols, site)
        drop = [c for c in drop if c not in extra]
    drop += list(v.get("drop", []))
    if v.get("drop_prefix"):
        drop += [c for c in train.columns if c.startswith(v["drop_prefix"])]
    return sorted(set(drop))


def _transform_target(y: np.ndarray, v: dict) -> np.ndarray:
    t = v.get("target", "raw")
    if t == "log1p":
        return np.log1p(y)
    if t == "sqrt":
        return np.sqrt(y)
    return y


def _inverse_target(p: np.ndarray, v: dict, smear: float = 1.0) -> np.ndarray:
    t = v.get("target", "raw")
    if t == "log1p":
        return np.expm1(p) * smear + (smear - 1.0)  # (1 + y) scaled by the smearing factor
    if t == "sqrt":
        return np.maximum(p, 0) ** 2
    return p


def _sample_weight(train: pd.DataFrame, anchor: pd.Timestamp, v: dict) -> np.ndarray | None:
    w = np.ones(len(train))
    used = False
    if v.get("halflife_years"):
        age = (anchor - pd.to_datetime(train["date"])).dt.days.to_numpy() / 365.25
        w *= 0.5 ** (age / v["halflife_years"])
        used = True
    if v.get("spike_weight"):
        w *= np.where(train["da_raw"].to_numpy(float) >= config.SPIKE_THRESHOLD, v["spike_weight"], 1.0)
        used = True
    return w if used else None


def training_and_test(point: dict, v: dict):
    ctx = context()
    ff = ctx["ff"]
    site, D = point["site"], pd.Timestamp(point["date"])
    A = D - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
    od, ov = ctx["obs"][site]
    feats = v.get("features", "base")
    if feats == "asof":
        tab = asof_table(site, v.get("asof_gapfill", False))
        train = tab[tab["_known_by"] <= A].copy()
        if len(train) < config.MIN_TRAINING_SAMPLES:
            return None
        ff_site = ff[ff["site"] == site].sort_values("date")
        er = env_row(ff_site, A)
        if er is None:
            return None
        r = er.to_dict()
        r.update(asof_persistence(od, ov, A, D, monday(D)))
        r["date"] = D
        test = pd.DataFrame([r])
    else:
        train = get_site_training_frame(ff, site, A, config.MIN_TRAINING_SAMPLES)
        if train is None:
            return None
        if v.get("train_rows") == "real":
            train = train[~train["_is_interpolated"]].copy()
            if len(train) < config.MIN_TRAINING_SAMPLES:
                return None
        test = get_site_anchor_row(ff, site, D, A, max_date_diff_days=28)
        if test is None:
            return None
        obs_df = pd.DataFrame({"date": od[od <= np.datetime64(A)], "da_raw": ov[od <= np.datetime64(A)]})
        test = recompute_test_row_persistence_features(test, obs_df, config.SPIKE_THRESHOLD)
        if feats == "refresh":
            wd, wv = weekly_obs_upto(od, ov, A)
            td = monday(D)
            for c, val in asof_persistence(wd, wv, A, td, td).items():
                test[c] = val
    if v.get("train_years"):
        train = train[train["date"] >= A - pd.Timedelta(days=int(365.25 * v["train_years"]))]
        if len(train) < config.MIN_TRAINING_SAMPLES:
            return None
    if v.get("add_obs_age"):
        # Age of the most recent measurement, as of each row's information cutoff.
        last_obs_date = pd.Series(pd.NaT, index=train.index)
        if feats == "asof":
            cut = train["date"] - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
        else:
            cut = train["date"] - pd.Timedelta(days=1)
        pos = np.searchsorted(od, cut.to_numpy(dtype="datetime64[ns]"), side="right")
        last_obs_date = pd.to_datetime(np.where(pos > 0, od[np.maximum(pos - 1, 0)], np.datetime64("NaT")))
        train["days_since_last_obs"] = (train["date"].to_numpy() - last_obs_date.to_numpy()) / np.timedelta64(1, "D")
        pos_t = int(np.searchsorted(od, np.datetime64(A), side="right"))
        test["days_since_last_obs"] = (D - pd.Timestamp(od[pos_t - 1])).days if pos_t else np.nan
    naive = get_last_known_raw_da(
        pd.DataFrame({"date": od, "da_raw": ov}), anchor_date=A, max_age_days=config.PERSISTENCE_MAX_DAYS)
    if naive is None:
        return None
    train = add_temporal_features(train)
    test = add_temporal_features(test)
    return train, test, naive, A


def predict_point(point: dict, v: dict) -> dict | None:
    site = point["site"]
    built = training_and_test(point, v)
    if built is None:
        return None
    train, test, naive, A = built
    drop = _drop_cols(train, site, v)
    try:
        transformer, X_train = create_transformer(train, drop)
        X_tr = transformer.fit_transform(X_train)
        X_te = transformer.transform(test.drop(columns=drop, errors="ignore").reindex(columns=X_train.columns, fill_value=0))
    except Exception:
        return None
    y = train["da_raw"].astype(float).to_numpy()
    yt = _transform_target(y, v)
    sw = _sample_weight(train, A, v)

    xgb_params = _xgb_params(site, v)
    if v.get("monotone", True) and config.USE_MONOTONIC_CONSTRAINTS:
        cmap = config.MONOTONIC_FEATURE_CONSTRAINTS
        mono = tuple(cmap.get(c, 0) for c in X_tr.columns)
        if any(mono):
            xgb_params["monotone_constraints"] = mono
    xgb = build_xgb_regressor(xgb_params)
    if "seed" in v:
        xgb.set_params(random_state=v["seed"])
    xgb.fit(X_tr, yt, sample_weight=sw)
    smear = 1.0
    if v.get("smear"):
        res = yt - xgb.predict(X_tr)
        smear = float(np.mean(np.exp(res)))
    xgb_raw = float(_inverse_target(xgb.predict(X_te), v, smear)[0])

    rf_raw = np.nan
    w_xgb, w_rf, _ = _weights(site, v)
    if not (v.get("skip_rf_unweighted") and w_rf == 0):
        rf = build_rf_regressor(_rf_params(site, v))
        if "seed" in v:
            rf.set_params(random_state=v["seed"])
        rf.fit(X_tr, yt, sample_weight=sw)
        rf_raw = float(_inverse_target(rf.predict(X_te), v)[0])
    lin = build_linear_regressor()
    lin.fit(X_tr, yt, sample_weight=sw)
    lin_raw = float(_inverse_target(lin.predict(X_te), v)[0])

    sq, sm = get_site_clip_params(site) if v.get("per_site", True) else (None, None)
    real_y = y[~train["_is_interpolated"].to_numpy(bool)]
    out = {
        "site": site, "date": pd.Timestamp(point["date"]), "anchor_date": A,
        "actual_da": float(point["actual_da"]), "naive_prediction": float(naive),
        "xgb_raw": xgb_raw, "rf_raw": rf_raw, "lin_raw": lin_raw,
        "w_xgb": w_xgb, "w_rf": w_rf,
        "clip_q_site": sq if sq is not None else config.PREDICTION_CLIP_Q, "clip_max_site": sm,
        "n_train": len(train), "n_train_real": int(len(real_y)),
    }
    for q in QUANTILES:
        out[f"yq{int(q * 100)}"] = float(np.quantile(y, q))
        out[f"yq{int(q * 100)}_real"] = float(np.quantile(real_y, q)) if len(real_y) else np.nan
    return out


def predict_pooled(points: list[dict], v: dict) -> list[dict]:
    """One XGBoost model per anchor date over all sites (site one-hot), XGB only."""
    ctx = context()
    ff = ctx["ff"]
    out = []
    for p in points:
        site, D = p["site"], pd.Timestamp(p["date"])
        A = D - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
        parts = []
        for s in ff["site"].unique():
            t = get_site_training_frame(ff, s, A, config.MIN_TRAINING_SAMPLES)
            if t is not None:
                parts.append(t)
        train = add_temporal_features(pd.concat(parts, ignore_index=True))
        built = training_and_test(p, {})
        if built is None:
            continue
        _, test, naive, _ = built
        sites = sorted(ff["site"].unique())
        for s in sites:
            train[f"site_{s}"] = (train["site"] == s).astype(float)
            test[f"site_{s}"] = float(site == s)
        drop = ["date", "site", "da_raw", "da", "_is_interpolated"] + list(config.ZERO_IMPORTANCE_FEATURES)
        transformer, X_train = create_transformer(train, drop)
        X_tr = transformer.fit_transform(X_train)
        X_te = transformer.transform(test.drop(columns=drop, errors="ignore").reindex(columns=X_train.columns, fill_value=0))
        y = train["da_raw"].astype(float).to_numpy()
        params = dict(config.XGB_REGRESSION_PARAMS)
        params["n_jobs"] = 1
        cmap = config.MONOTONIC_FEATURE_CONSTRAINTS
        params["monotone_constraints"] = tuple(cmap.get(c, 0) for c in X_tr.columns)
        xgb = build_xgb_regressor(params)
        xgb.fit(X_tr, y)
        site_y = train.loc[train["site"] == site, "da_raw"].to_numpy(float)
        sq, sm = get_site_clip_params(site)
        rec = {
            "site": site, "date": D, "anchor_date": A, "actual_da": float(p["actual_da"]),
            "naive_prediction": float(naive), "xgb_raw": float(xgb.predict(X_te)[0]),
            "rf_raw": np.nan, "lin_raw": np.nan, "w_xgb": 1.0, "w_rf": 0.0,
            "clip_q_site": sq if sq is not None else config.PREDICTION_CLIP_Q, "clip_max_site": sm,
            "n_train": len(train), "n_train_real": int(np.isfinite(site_y).sum()),
        }
        for q in QUANTILES:
            rec[f"yq{int(q * 100)}"] = float(np.quantile(site_y, q))
            rec[f"yq{int(q * 100)}_real"] = np.nan
        out.append(rec)
    return out


def spike_probability(point: dict, v: dict) -> dict | None:
    """Pooled spike classifier as in RawForecastEngine (classifier-only variants).

    ``cls_features``: "base" (engine), "refresh" or "asof" (training and test rows
    built as of target - 7 days). ``cls_safe_fixed`` applies the documented
    safe-baseline filter on the *unscaled* da_raw_prev_obs_1; the engine applies it
    to the MinMax-scaled column, where every value is < 20, so it keeps all rows.
    """
    from sklearn.utils.class_weight import compute_class_weight
    from xgboost import XGBClassifier
    ctx = context()
    ff = ctx["ff"]
    site, D = point["site"], pd.Timestamp(point["date"])
    A = D - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
    feats = v.get("cls_features", "base")
    built = training_and_test(point, {"features": feats})
    if built is None:
        return None
    _, test, naive, _ = built
    if feats == "asof":
        parts = [asof_table(s, False) for s in sorted(ff["site"].unique())]
        all_train = pd.concat(parts, ignore_index=True)
        all_train = all_train[all_train["_known_by"] <= A].copy()
    else:
        all_train = ff[week_complete_by(ff["date"], A) & ff["da_raw"].notna()].copy()
    if len(all_train) < 30:
        return None
    all_train = add_temporal_features(all_train)
    y = all_train["da_raw"].astype(float)
    drop = ["date", "site", "da_raw", "da", "_is_interpolated", "_known_by"] + list(config.ZERO_IMPORTANCE_FEATURES)
    drop += list(v.get("cls_drop", []))
    transformer, X = create_transformer(all_train, drop)
    cols = list(X.columns)
    Xs = transformer.fit_transform(X)
    thr = config.SPIKE_THRESHOLD
    if v.get("cls_safe_fixed"):
        mask = all_train["da_raw_prev_obs_1"].fillna(0).to_numpy() < thr
    else:
        mask = (Xs["da_raw_prev_obs_1"].fillna(0) < thr).to_numpy()
    leaky = [c for c in ("last_observed_da_raw", "weeks_since_last_spike") if c in Xs.columns]
    X_safe = Xs.loc[mask].drop(columns=leaky)
    y_safe = (y.to_numpy()[mask] >= thr).astype(int)
    if len(y_safe) < 10 or len(np.unique(y_safe)) < 2:
        return None
    classes = np.unique(y_safe)
    cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_safe)))
    model = XGBClassifier(**dict(config.SPIKE_CLASSIFIER_PARAMS), random_state=config.MODEL_SEED,
                          verbosity=0, use_label_encoder=False)
    model.fit(X_safe, y_safe, sample_weight=np.array([cw[c] for c in y_safe]))
    X_te = transformer.transform(test.reindex(columns=cols)).reindex(columns=list(X_safe.columns), fill_value=0)
    proba = model.predict_proba(X_te)
    return {"site": site, "date": D, "anchor_date": A, "actual_da": float(point["actual_da"]),
            "naive_prediction": float(naive), "spike_prob": float(proba[0, 1]) if proba.shape[1] > 1 else 0.0,
            "n_cls_train": int(len(y_safe))}


def _run_chunk(points: list[dict], variant: dict) -> list[dict]:
    import warnings
    warnings.simplefilter("ignore")
    if variant.get("pooled"):
        return predict_pooled(points, variant)
    if variant.get("classifier_only"):
        return [r for r in (spike_probability(p, variant) for p in points) if r is not None]
    res = []
    for p in points:
        r = predict_point(p, variant)
        if r is not None:
            res.append(r)
    return res


# ---------------------------------------------------------------------------
# Test points and driver
# ---------------------------------------------------------------------------

def test_points(window: str) -> pd.DataFrame:
    start, end, path = WINDOWS[window]
    if path is not None:
        df = pd.read_parquet(ROOT / path)
        df["date"] = pd.to_datetime(df["date"])
        return df[["site", "date", "actual_da", "predicted_da", "spike_probability", "spike_alert"]]
    # Same history filter as RawForecastEngine.run_retrospective_evaluation
    ctx = context()
    raw, ff = ctx["raw"], ctx["ff"]
    totals = raw.groupby("site").size().to_dict()
    cand = raw[(raw["date"] >= start) & (raw["date"] < end)]
    keep = []
    for _, r in cand.iterrows():
        a = r["date"] - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
        need = max(int(np.ceil(config.HISTORY_REQUIREMENT_FRACTION * totals[r["site"]])), config.MIN_TRAINING_SAMPLES)
        if ((raw["site"] == r["site"]) & (raw["date"] <= a)).sum() < need:
            continue
        hist = ff[(ff["site"] == r["site"]) & (ff["date"] <= a) & ff["da_raw"].notna()]
        if len(hist) >= config.MIN_TRAINING_SAMPLES:
            keep.append(r)
    df = pd.DataFrame(keep).rename(columns={"da_raw": "actual_da"})
    df["predicted_da"] = np.nan
    df["spike_probability"] = np.nan
    df["spike_alert"] = np.nan
    return df[["site", "date", "actual_da", "predicted_da", "spike_probability", "spike_alert"]]


def run_variant(name: str, window: str, n_jobs: int, limit: int | None, outdir: Path) -> pd.DataFrame:
    from joblib import Parallel, delayed
    v = VARIANTS[name]
    pts = test_points(window).sort_values(["site", "date"]).reset_index(drop=True)
    if limit:
        pts = pts.groupby("site", group_keys=False).head(max(1, limit // pts["site"].nunique()))
    recs = pts[["site", "date", "actual_da"]].to_dict("records")
    n_chunks = max(1, min(len(recs), (n_jobs if n_jobs > 0 else os.cpu_count()) * 4))
    chunks = [recs[i::n_chunks] for i in range(n_chunks)]
    t0 = time.time()
    if n_jobs == 1:
        parts = [_run_chunk(c, v) for c in chunks]
    else:
        parts = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_run_chunk)(c, v) for c in chunks)
    res = pd.DataFrame([r for part in parts for r in part])
    res = res.merge(pts[["site", "date", "predicted_da", "spike_probability", "spike_alert"]]
                    .rename(columns={"predicted_da": "committed_ensemble"}), on=["site", "date"], how="left")
    res = res.sort_values(["date", "site"]).reset_index(drop=True)
    res.attrs["variant"] = json.dumps(v)
    outdir.mkdir(parents=True, exist_ok=True)
    res.to_parquet(outdir / f"{window}__{name}.parquet", index=False)
    print(f"{window:8s} {name:18s} n={len(res):4d}/{len(pts):4d}  {time.time() - t0:6.1f}s", flush=True)
    return res


def ensemble(res: pd.DataFrame) -> np.ndarray:
    """Committed post-processing: max(0, .), clip at the site quantile of all training targets."""
    def clip(col):
        v = np.maximum(res[col].to_numpy(float), 0.0)
        bound = np.array([res.loc[i, f"yq{int(round(q * 100))}"] for i, q in enumerate(res["clip_q_site"])])
        v = np.minimum(v, bound)
        cm = res["clip_max_site"].to_numpy(float)
        return np.where(np.isfinite(cm), np.minimum(v, cm), v)
    rf = clip("rf_raw") if res["rf_raw"].notna().all() else clip("xgb_raw")
    return res["w_xgb"].to_numpy() * clip("xgb_raw") + res["w_rf"].to_numpy() * rf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="val", choices=list(WINDOWS))
    ap.add_argument("--variants", default="baseline")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None, help="at most this many points (spread over sites)")
    ap.add_argument("--out-dir", default=os.environ.get("EXP_OUT", "results/experiments/raw"))
    ap.add_argument("--verify", action="store_true", help="compare baseline with the committed ensemble")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    names = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
    for name in names:
        if args.skip_existing and (Path(args.out_dir) / f"{args.window}__{name}.parquet").exists():
            print(f"{args.window:8s} {name:18s} exists, skipped", flush=True)
            continue
        res = run_variant(name, args.window, args.n_jobs, args.limit, Path(args.out_dir))
        if args.verify and name == "baseline":
            diff = np.abs(ensemble(res) - res["committed_ensemble"].to_numpy(float))
            print(f"  verify: max |harness - committed| = {np.nanmax(diff):.3e} over {len(res)} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

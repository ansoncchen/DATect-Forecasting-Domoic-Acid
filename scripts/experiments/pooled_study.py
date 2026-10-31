#!/usr/bin/env python3
"""Pooled single-model comparison, declared in docs/FORECAST_EXPERIMENTS.md §3b.

One model per family across all sites; no per-site anything. Training rows are real
measurements built as of (measurement date - 7 days); every candidate sees the same
rows and features. A pooled fit depends only on the anchor date, so forecast points
sharing an anchor share one fit per family.

For each test point and each (feature set, family) it records
``p_<fs>_<fam>`` (onset classifier: P(DA >= 20), trained on rows with last known < 20)
and ``lvl_<fs>_<fam>`` (level regressor, trained on all rows, floored at 0).

    python scripts/experiments/pooled_study.py --window val --out-dir results/experiments/raw
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness as H  # noqa: E402  (sets sys.path / cwd to the repo root)
import config  # noqa: E402
from forecasting.feature_utils import add_temporal_features, create_transformer  # noqa: E402
from forecasting.raw_model_factory import build_rf_regressor, build_xgb_regressor  # noqa: E402

FAMILIES = ("linear", "rf", "xgb")
FEATURE_SETS = {"F1": {}, "F2": {"regional": True, "pn_asof": True}}
DROP = ["date", "site", "da_raw", "da", "_is_interpolated", "_known_by"]
_ROWS: dict = {}


def sites() -> list[str]:
    return sorted(H.context()["obs"])


def one_hot(df: pd.DataFrame) -> pd.DataFrame:
    for s in sites():
        df[f"site_{s}"] = (df["site"] == s).astype(float)
    return df


def pooled_rows(fs: str) -> pd.DataFrame:
    """All sites' horizon-consistent real-measurement rows (cached per process)."""
    if fs in _ROWS:
        return _ROWS[fs]
    opts = FEATURE_SETS[fs]
    parts = []
    for s in sites():
        t = H.asof_table(s, False).copy()
        t["site"] = s
        if opts:
            extra = H.context_features(s, t["date"] - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS), opts)
            extra.index = t.index
            t = pd.concat([t, extra], axis=1)
        parts.append(t)
    df = add_temporal_features(one_hot(pd.concat(parts, ignore_index=True)))
    _ROWS[fs] = df
    return df


def test_row(point: dict, fs: str) -> tuple[pd.DataFrame, float] | None:
    ctx = H.context()
    site, D = point["site"], pd.Timestamp(point["date"])
    A = D - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
    ff_site = ctx["ff"][ctx["ff"]["site"] == site].sort_values("date")
    er = H.env_row(ff_site, A)
    od, ov = ctx["obs"][site]
    n_known = int(np.searchsorted(od, np.datetime64(A), side="right"))
    if er is None or n_known == 0:
        return None
    r = er.to_dict()
    r.update(H.asof_persistence(od, ov, A, D, H.monday(D)))
    r.update({"date": D, "site": site})
    if FEATURE_SETS[fs]:
        r.update(H.context_features(site, [A], FEATURE_SETS[fs]).iloc[0].to_dict())
    return add_temporal_features(one_hot(pd.DataFrame([r]))), float(ov[n_known - 1])


def _balanced(y: np.ndarray) -> np.ndarray:
    w = np.ones(len(y))
    for c in (0, 1):
        n = (y == c).sum()
        if n:
            w[y == c] = len(y) / (2.0 * n)
    return w


def fit_predict(fam: str, X_tr, y_lvl, X_on, y_on, X_te) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from xgboost import XGBClassifier
    seed = config.MODEL_SEED
    if fam == "linear":
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
        reg = Ridge(alpha=config.LINEAR_REGRESSION_ALPHA, random_state=seed)
        clf.fit(X_on, y_on)
        reg.fit(X_tr, y_lvl)
    elif fam == "rf":
        clf = RandomForestClassifier(**config.RF_REGRESSION_PARAMS, class_weight="balanced_subsample",
                                     n_jobs=1, random_state=seed)
        reg = build_rf_regressor()
        clf.fit(X_on, y_on)
        reg.fit(X_tr, y_lvl)
    else:
        clf = XGBClassifier(**dict(config.SPIKE_CLASSIFIER_PARAMS), random_state=seed, verbosity=0, n_jobs=1)
        reg = build_xgb_regressor({"n_jobs": 1})
        clf.fit(X_on, y_on, sample_weight=_balanced(y_on))
        reg.fit(X_tr, y_lvl)
    return clf.predict_proba(X_te)[:, 1], np.maximum(reg.predict(X_te), 0.0)


def run_anchor(anchor: pd.Timestamp, points: list[dict]) -> list[dict]:
    import warnings
    warnings.simplefilter("ignore")
    out = {(p["site"], pd.Timestamp(p["date"])): {
        "site": p["site"], "date": pd.Timestamp(p["date"]), "anchor_date": anchor,
        "actual_da": float(p["actual_da"])} for p in points}
    for fs in FEATURE_SETS:
        rows = pooled_rows(fs)
        train = rows[rows["_known_by"] <= anchor]
        built = [(p, test_row(p, fs)) for p in points]
        built = [(p, b) for p, b in built if b is not None]
        if len(train) < 30 or not built:
            continue
        test = pd.concat([b[0] for _, b in built], ignore_index=True)
        drop = DROP + list(config.ZERO_IMPORTANCE_FEATURES)
        transformer, X = create_transformer(train, drop)
        X_tr = transformer.fit_transform(X)
        X_te = transformer.transform(test.drop(columns=drop, errors="ignore").reindex(columns=X.columns, fill_value=0))
        y_lvl = train["da_raw"].to_numpy(float)
        cand = train["last_observed_da_raw"].fillna(0).to_numpy(float) < config.SPIKE_THRESHOLD
        y_on = (y_lvl[cand] >= config.SPIKE_THRESHOLD).astype(int)
        for fam in FAMILIES:
            p_on, lvl = fit_predict(fam, X_tr, y_lvl, X_tr[cand], y_on, X_te)
            for i, (p, b) in enumerate(built):
                rec = out[(p["site"], pd.Timestamp(p["date"]))]
                rec[f"p_{fs}_{fam}"] = float(p_on[i])
                rec[f"lvl_{fs}_{fam}"] = float(lvl[i])
                rec["naive_prediction"] = b[1]
                rec["n_train"] = int(len(train))
                rec["n_train_onset"] = int(cand.sum())
    return [r for r in out.values() if "naive_prediction" in r]


def _run_groups(groups: list[tuple[pd.Timestamp, list[dict]]]) -> list[dict]:
    res = []
    for a, pts in groups:
        res.extend(run_anchor(a, pts))
    return res


def main() -> int:
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="val", choices=list(H.WINDOWS))
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None, help="at most this many anchor dates")
    ap.add_argument("--out-dir", default=os.environ.get("EXP_OUT", "results/experiments/raw"))
    args = ap.parse_args()
    pts = H.test_points(args.window)
    pts["anchor"] = pts["date"] - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
    groups = [(a, g[["site", "date", "actual_da"]].to_dict("records")) for a, g in pts.groupby("anchor")]
    if args.limit:
        groups = groups[:: max(1, len(groups) // args.limit)][: args.limit]
    n_workers = args.n_jobs if args.n_jobs > 0 else os.cpu_count()
    chunks = [groups[i::n_workers * 2] for i in range(n_workers * 2)]
    t0 = time.time()
    parts = Parallel(n_jobs=args.n_jobs, backend="loky")(delayed(_run_groups)(c) for c in chunks if c)
    res = pd.DataFrame([r for part in parts for r in part]).sort_values(["date", "site"]).reset_index(drop=True)
    committed = pts[["site", "date", "predicted_da", "spike_probability", "spike_alert"]]
    res = res.merge(committed.rename(columns={"predicted_da": "committed_ensemble"}), on=["site", "date"], how="left")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out / f"{args.window}__pooled_study.parquet", index=False)
    print(f"{args.window} pooled_study n={len(res)}/{len(pts)} anchors={len(groups)} {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Onset round 2 (docs/FORECAST_EXPERIMENTS.md §6): spike-classifier variants.

Refits the engine's pooled spike classifier (``RawForecastEngine._run_single_raw_validation``)
with one change at a time, on every real measurement of the committed all-years
chronological run dated in [--start, --end). The classifier depends only on the anchor
date, so the points that share an anchor share one fit. ``baseline`` reproduces the
committed ``spike_probability`` (checked by ``--verify``).

Added features are computed as of each row's information cutoff: the day before its week
for a weekly training row, the measurement date minus 7 days for a horizon-consistent
(``asof``) training row, and the anchor for a forecast row.

    python scripts/experiments/onset_study.py --variants baseline --limit 20 --verify
    python scripts/experiments/onset_study.py --variants all --start 2015-01-01 --end 2022-01-01
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
os.chdir(ROOT)

import config  # noqa: E402
import transport as T  # noqa: E402
from forecasting.feature_utils import add_temporal_features, create_transformer  # noqa: E402
from forecasting.raw_data_forecaster import (  # noqa: E402
    get_site_anchor_row, recompute_test_row_persistence_features, week_complete_by,
)

COMMITTED = "results/chronological/chronological_regression_ensemble_20030101_20240101.parquet"
THR = float(config.SPIKE_THRESHOLD)
H = int(config.FORECAST_HORIZON_DAYS)

REGIONS = {
    "WA": ["Kalaloch", "Quinault", "Copalis", "Twin Harbors", "Long Beach"],
    "N_OR": ["Clatsop Beach", "Cannon Beach"],
    "S_OR": ["Newport", "Coos Bay", "Gold Beach"],
}
SITE_REGION = {s: r for r, ss in REGIONS.items() for s in ss}
REGIONAL_COLS = ["nb_max_14", "nb_max_28", "nb_last_mean", "nb_n_hi_14", "nb_rise_14",
                 "coast_max_14", "coast_n_hi_14"]
PN_COLS = ["pn_last_log", "pn_max14_log", "pn_age_days", "pn_nb_max14_log", "pn_rise_log"]
TREND_COLS = ["da_slope_3", "weeks_since_obs", "weeks_since_detect", "weeks_since_ge5"]
REGION_COLS = [f"region_{r}" for r in REGIONS]
DETECT, ELEVATED = 1.0, 5.0  # "above the detection limit" and the [5, 20) category
# §7.2 plumbing checks for the null result, never candidates (see control_columns)
CONTROL_NOISE_COLS = [f"ctl_noise{i}" for i in range(7)]

# Each variant changes one thing relative to the engine's classifier (§6 hypotheses).
VARIANTS: dict[str, dict] = {
    "baseline": {},
    "seed43": {"seed": 43},
    "seed44": {"seed": 44},
    "regional": {"extra": REGIONAL_COLS},                                  # A1
    "pn_raw": {"extra": PN_COLS},                                          # A2
    "asof": {"rows": "asof"},                                              # A3
    "asof+regional+pn_raw": {"rows": "asof", "extra": REGIONAL_COLS + PN_COLS},  # A4
    "trend": {"extra": TREND_COLS},                                        # B1
    "label_2w": {"label": "2w"},                                           # C1
    "label_10": {"label": "10"},                                           # C2
    "no_weight": {"weight": "none"},                                       # D1
    "sqrt_weight": {"weight": "sqrt"},                                     # D2
    "region": {"extra": REGION_COLS},                                      # D3
    # transport-trigger round (§7); need DATECT_TT_WIND / DATECT_TT_NEMO
    "wind": {"extra": T.WIND_COLS},                                        # T1
    "mooring": {"extra": T.NEMO_COLS},                                     # T2
    "wind+mooring": {"extra": T.WIND_COLS + T.NEMO_COLS},                  # T3
    "wind+interactions": {"extra": T.WIND_COLS + T.INTERACTION_COLS},      # T4
    "pn_crash": {"extra": T.PN_CRASH_COLS},                                # T5 (§7.1)
    "sat_lag": {"extra": T.SAT_COLS},                                      # T6 (§7.1)
    "sat_lag+wind": {"extra": T.WIND_COLS + T.SAT_COLS + ["sat_x_reversal"]},  # T7 (§7.1)
    "control_oracle": {"extra": ["ctl_oracle"]},
    "control_noise": {"extra": CONTROL_NOISE_COLS},
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
        obs = {s: (g["date"].to_numpy("datetime64[ns]"), g["da_raw"].to_numpy(float))
               for s, g in raw.groupby("site")}
        _CTX = {"eng": eng, "ff": ff, "obs": obs, "pn": load_raw_pn(), "tables": {}, "cf": {}}
    return _CTX


def load_raw_pn() -> dict:
    """Raw PN cell counts per site (one value per site-day, the max), sorted by date."""
    from forecasting.measurements import parse_dates, parse_pn_values
    out = {}
    for key, path in config.ORIGINAL_PN_FILES.items():
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        col = [c for c in d.columns if "pseudo" in str(c).lower() and "nitzschia" in str(c).lower()]
        if "Date" not in d.columns or len(col) != 1:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dates = parse_dates(d["Date"])
        df = pd.DataFrame({"date": dates, "pn": parse_pn_values(d[col[0]])}).dropna()
        df = df.groupby("date", as_index=False)["pn"].max().sort_values("date")
        site = key.replace("-pn", "").replace("-", " ").title()
        out[site] = (df["date"].to_numpy("datetime64[ns]"), df["pn"].to_numpy(float))
    return out


def monday(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts - pd.Timedelta(days=ts.weekday())


# ---------------------------------------------------------------------------
# Added features, as of an information cutoff
# ---------------------------------------------------------------------------

def _window(dates: np.ndarray, vals: np.ndarray, cutoff: np.datetime64, days: int):
    lo = np.searchsorted(dates, cutoff - np.timedelta64(days, "D"), side="right")
    hi = np.searchsorted(dates, cutoff, side="right")
    return vals[lo:hi], hi


def context_features(site: str, cutoff) -> dict:
    """Cross-site DA (REGIONAL_COLS) and raw-PN (PN_COLS) features from samples dated
    on or before ``cutoff``. "nb_" = other sites in the coastal group, "coast_" = every
    other site: max DA in 14/28 days, mean of each neighbour's latest value (if at most
    28 days old), neighbours with a value >= 20 in 14 days, and the largest
    latest-minus-previous rise among neighbours sampled in the last 14 days. PN: own
    last count (log1p) and its age in days, own 14-day max, the coastal group's 14-day
    max, and the change between the last two own samples (log1p)."""
    ctx = context()
    c = np.datetime64(pd.Timestamp(cutoff), "ns")
    key = (site, c)
    if key in ctx["cf"]:
        return ctx["cf"][key]
    f: dict = {}
    nbrs = [s for s in REGIONS[SITE_REGION[site]] if s != site]
    others = [s for s in ctx["obs"] if s != site]
    for prefix, group, windows in (("nb", nbrs, (14, 28)), ("coast", others, (14,))):
        mx = {w: -np.inf for w in windows}
        n_hi, lasts, rise = 0, [], -np.inf
        for s in group:
            od, ov = ctx["obs"][s]
            for w in windows:
                vals, _ = _window(od, ov, c, w)
                if len(vals):
                    mx[w] = max(mx[w], vals.max())
            vals14, hi = _window(od, ov, c, 14)
            n_hi += int(len(vals14) > 0 and vals14.max() >= THR)
            if hi > 0 and (c - od[hi - 1]) <= np.timedelta64(28, "D"):
                lasts.append(ov[hi - 1])
                if len(vals14) and hi > 1:
                    rise = max(rise, ov[hi - 1] - ov[hi - 2])
        for w in windows:
            f[f"{prefix}_max_{w}"] = mx[w] if np.isfinite(mx[w]) else np.nan
        f[f"{prefix}_n_hi_14"] = n_hi
        if prefix == "nb":
            f["nb_last_mean"] = float(np.mean(lasts)) if lasts else np.nan
            f["nb_rise_14"] = rise if np.isfinite(rise) else np.nan
    f.update({k: np.nan for k in PN_COLS})
    if site in ctx["pn"]:
        pdates, pv = ctx["pn"][site]
        hi = np.searchsorted(pdates, c, side="right")
        if hi > 0:
            f["pn_last_log"] = np.log1p(pv[hi - 1])
            f["pn_age_days"] = (c - pdates[hi - 1]) / np.timedelta64(1, "D")
            if hi > 1:
                f["pn_rise_log"] = np.log1p(pv[hi - 1]) - np.log1p(pv[hi - 2])
        vals, _ = _window(pdates, pv, c, 14)
        if len(vals):
            f["pn_max14_log"] = np.log1p(vals.max())
    mx = -np.inf
    for s in REGIONS[SITE_REGION[site]]:
        if s in ctx["pn"]:
            vals, _ = _window(*ctx["pn"][s], c, 14)
            if len(vals):
                mx = max(mx, vals.max())
    f["pn_nb_max14_log"] = np.log1p(mx) if np.isfinite(mx) else np.nan
    ctx["cf"][key] = f
    return f


def trend_features(weeks: np.ndarray, values: np.ndarray, target_week) -> dict:
    """TREND_COLS from weekly maxima dated (by Monday) before ``target_week``: the
    least-squares slope of the last three (ug/g per week) and the weeks since the last
    measured week, the last value > 1 and the last value >= 5 (999 if none)."""
    f = {"da_slope_3": np.nan, "weeks_since_obs": np.nan,
         "weeks_since_detect": 999.0, "weeks_since_ge5": 999.0}
    if len(values) == 0:
        return f
    ago = (np.datetime64(pd.Timestamp(target_week), "ns") - weeks) / np.timedelta64(7, "D")
    f["weeks_since_obs"] = float(ago[-1])
    for col, mask in (("weeks_since_detect", values > DETECT), ("weeks_since_ge5", values >= ELEVATED)):
        if mask.any():
            f[col] = float(ago[mask][-1])
    if len(values) >= 2:
        x, y = -ago[-3:], values[-3:]
        f["da_slope_3"] = float(np.polyfit(x, y, 1)[0])
    return f


def weekly_maxima(od: np.ndarray, ov: np.ndarray, cutoff, before_week) -> tuple[np.ndarray, np.ndarray]:
    """Weekly maxima (dated by Monday) of samples dated <= cutoff in weeks before ``before_week``,
    as recompute_test_row_persistence_features builds them."""
    m = od <= np.datetime64(pd.Timestamp(cutoff), "ns")
    if not m.any():
        return od[:0], ov[:0]
    s = pd.Series(ov[m], index=pd.DatetimeIndex(od[m]))
    wk = s.groupby(s.index - pd.to_timedelta(s.index.weekday, unit="D")).max()
    wk = wk[wk.index < pd.Timestamp(before_week)]
    return wk.index.to_numpy("datetime64[ns]"), wk.to_numpy(float)


def extra_columns(site: str, cutoff, weeks: np.ndarray, values: np.ndarray, target_week) -> dict:
    f = dict(context_features(site, cutoff))
    f.update(trend_features(weeks, values, target_week))
    f.update({f"region_{r}": float(SITE_REGION[site] == r) for r in REGIONS})
    return f


# ---------------------------------------------------------------------------
# Training tables (one per process)
# ---------------------------------------------------------------------------

def transport_columns(site: str, cutoff) -> dict:
    """§7 features as of ``cutoff`` (wind needs DATECT_TT_WIND, mooring DATECT_TT_NEMO;
    absent inputs give no columns). The interactions use the own-site PN, the neighbours'
    14-day max DA and the last DA sample, all as of the same cutoff."""
    f: dict = {}
    if os.environ.get("DATECT_TT_WIND"):
        w = T.wind_features(cutoff)
        cf = context_features(site, cutoff)
        od, ov = context()["obs"][site]
        hi = np.searchsorted(od, np.datetime64(pd.Timestamp(cutoff), "ns"), side="right")
        f.update(w)
        f.update(T.interaction_features(w, cf["pn_last_log"], cf["nb_max_14"], ov[hi - 1] if hi else np.nan))
    if os.environ.get("DATECT_TT_NEMO"):
        f.update(T.nemo_features(cutoff))
    ctx = context()
    if site in ctx["pn"]:
        f.update(T.pn_crash_features(*ctx["pn"][site], cutoff))
    else:
        f.update(dict.fromkeys(T.PN_CRASH_COLS, np.nan))
    sat = T.sat_features(site, cutoff)
    f.update(sat)
    rev = f.get("wind_reversal_7d", np.nan)
    f["sat_x_reversal"] = rev * sat["sat_max_2_8w"] if np.isfinite(rev) else np.nan
    return f


def control_columns(site: str, day, measured: float) -> dict:
    """§7.2 plumbing checks. ``ctl_oracle`` deliberately leaks the row's own spike label
    (flipped on 20% of rows), so an extra column that reaches the model must show a large
    gain; ``ctl_noise*`` are random. Seeded per site-day so every worker agrees."""
    u = np.random.default_rng(zlib.crc32(f"{site}|{pd.Timestamp(day).date()}".encode())).random(8)
    spike = float(measured >= THR)
    return {**{c: float(x) for c, x in zip(CONTROL_NOISE_COLS, u)},
            "ctl_oracle": 1.0 - spike if u[7] < 0.2 else spike}


def lead_cutoff(site: str, week) -> pd.Timestamp:
    """Training rows' §7 cutoff: the sample date of the week's maximum minus 7 days (the
    lead a forecast row has); the earliest such date when several days tie."""
    od, ov = context()["obs"][site]
    lo = np.searchsorted(od, np.datetime64(pd.Timestamp(week), "ns"), side="left")
    hi = np.searchsorted(od, np.datetime64(pd.Timestamp(week) + pd.Timedelta(days=7), "ns"), side="left")
    day = pd.Timestamp(od[lo + int(np.argmax(ov[lo:hi]))]) if hi > lo else pd.Timestamp(week)
    return day - pd.Timedelta(days=H)


def weekly_table() -> pd.DataFrame:
    """The engine's pooled training rows (real weekly rows of the feature frame, in its
    order) with every added column, the next week's value and the label horizon."""
    ctx = context()
    if "weekly" in ctx["tables"]:
        return ctx["tables"]["weekly"]
    ff = ctx["ff"]
    real = ff[ff["da_raw"].notna()].copy()
    extras = {}
    for site, g in real.groupby("site", sort=False):
        g = g.sort_values("date")
        weeks = g["date"].to_numpy("datetime64[ns]")
        vals = g["da_raw"].to_numpy(float)
        nxt = dict(zip(g["date"], g["da_raw"]))
        for i, (idx, w) in enumerate(zip(g.index, g["date"])):
            f = extra_columns(site, w - pd.Timedelta(days=1), weeks[:i], vals[:i], w)
            f["_da_next_week"] = nxt.get(w + pd.Timedelta(days=7), np.nan)
            f.update(transport_columns(site, lead_cutoff(site, w)))
            f.update(control_columns(site, w, vals[i]))
            extras[idx] = f
    ext = pd.DataFrame.from_dict(extras, orient="index").loc[real.index]
    tab = pd.concat([real, ext], axis=1)
    ctx["tables"]["weekly"] = tab
    return tab


def asof_table() -> pd.DataFrame:
    """Horizon-consistent training rows: one per real measurement (site-day max), built
    like a forecast row for that day: environmental row whose week ends by day - 7,
    persistence features from samples dated <= day - 7 (the engine's own rebuild)."""
    ctx = context()
    if "asof" in ctx["tables"]:
        return ctx["tables"]["asof"]
    ff, rows = ctx["ff"], []
    for site, (od, ov) in ctx["obs"].items():
        ff_site = ff[ff["site"] == site].sort_values("date")
        for d, y in zip(od, ov):
            d = pd.Timestamp(d)
            a = d - pd.Timedelta(days=H)
            row = get_site_anchor_row(ff_site, site, d, a, max_date_diff_days=28)
            if row is None:
                continue
            m = od <= np.datetime64(a, "ns")
            if not m.any():
                continue
            hist = pd.DataFrame({"date": pd.DatetimeIndex(od[m]), "da_raw": ov[m]})
            row = recompute_test_row_persistence_features(row, hist, THR)
            weeks, vals = weekly_maxima(od, ov, a, monday(d))
            r = row.iloc[0].to_dict()
            r.update(extra_columns(site, a, weeks, vals, monday(d)))
            r.update({"date": d, "da_raw": float(y), "_known_by": d, "_da_next_week": np.nan})
            rows.append(r)
    tab = pd.DataFrame(rows).sort_values(["date", "site"], kind="stable").reset_index(drop=True)
    ctx["tables"]["asof"] = tab
    return tab


# ---------------------------------------------------------------------------
# One anchor
# ---------------------------------------------------------------------------

def test_rows(points: list[dict], A: pd.Timestamp) -> list[tuple[dict, pd.DataFrame]]:
    """Forecast rows exactly as the engine builds them, plus the added columns as of A."""
    ctx = context()
    out = []
    for p in points:
        site, D = p["site"], pd.Timestamp(p["date"])
        row = get_site_anchor_row(ctx["ff"], site, D, A, max_date_diff_days=28)
        if row is None:
            continue
        row = recompute_test_row_persistence_features(row, ctx["eng"]._observations_up_to(site, A), THR)
        row = add_temporal_features(row)
        od, ov = ctx["obs"][site]
        weeks, vals = weekly_maxima(od, ov, A, monday(D))
        for k, val in {**extra_columns(site, A, weeks, vals, monday(D)), **transport_columns(site, A),
                       **control_columns(site, D, p["actual_da"])}.items():
            row[k] = val
        out.append((p, row))
    return out


def run_anchor(A: pd.Timestamp, points: list[dict], v: dict) -> list[dict]:
    from sklearn.utils.class_weight import compute_class_weight
    from xgboost import XGBClassifier
    ctx = context()
    base_cols = list(ctx["ff"].columns)
    extra = list(v.get("extra", []))
    if v.get("rows") == "asof":
        tab = asof_table()
        train = tab[tab["_known_by"] <= A]
    else:
        tab = weekly_table()
        if v.get("label") == "2w":  # both weeks complete by the anchor
            train = tab[week_complete_by(tab["date"] + pd.Timedelta(days=7), A)]
        else:
            train = tab[week_complete_by(tab["date"], A)]
    train = train[[c for c in base_cols if c in train.columns] + extra + ["_da_next_week"]].copy()
    if len(train) < 30:
        return []
    train = add_temporal_features(train)
    y_raw = train["da_raw"].astype(float).to_numpy()
    if v.get("label") == "2w":
        y = (np.fmax(y_raw, train["_da_next_week"].astype(float).to_numpy()) >= THR).astype(int)
    elif v.get("label") == "10":
        y = (y_raw >= 10.0).astype(int)
    else:
        y = (y_raw >= THR).astype(int)
    drop = (["date", "site", "da_raw", "da", "_is_interpolated", "_known_by", "_da_next_week"]
            + list(config.ZERO_IMPORTANCE_FEATURES))
    transformer, X = create_transformer(train, drop)
    columns = list(X.columns)
    Xs = transformer.fit_transform(X)
    leaky = [c for c in ("last_observed_da_raw", "weeks_since_last_spike", "distance_to_threshold")
             if c in Xs.columns]
    Xs = Xs.drop(columns=leaky)
    if len(y) < 10 or len(np.unique(y)) < 2:
        return []
    classes = np.unique(y)
    weight = v.get("weight", "balanced")
    if weight == "none":
        sw = None
    else:
        cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y)))
        if weight == "sqrt":
            cw = {0: 1.0, 1: float(np.sqrt((y == 0).sum() / (y == 1).sum()))}
        sw = np.array([cw[c] for c in y])
    model = XGBClassifier(**dict(config.SPIKE_CLASSIFIER_PARAMS),
                          random_state=v.get("seed", config.MODEL_SEED), verbosity=0,
                          use_label_encoder=False)
    model.fit(Xs, y, sample_weight=sw)
    out = []
    for p, row in test_rows(points, A):
        X_te = transformer.transform(row.reindex(columns=columns))
        X_te = X_te.reindex(columns=list(Xs.columns), fill_value=0)
        proba = model.predict_proba(X_te)
        out.append({"site": p["site"], "date": pd.Timestamp(p["date"]), "anchor_date": A,
                    "spike_prob": float(proba[0, 1]) if proba.shape[1] > 1 else 0.0,
                    "n_train": int(len(y)), "n_pos": int(y.sum())})
    return out


def _run_chunk(groups: list[tuple[pd.Timestamp, list[dict]]], v: dict) -> list[dict]:
    warnings.simplefilter("ignore")
    res = []
    for A, pts in groups:
        res.extend(run_anchor(A, pts, v))
    return res


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def test_points(start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / COMMITTED)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start) & (df["date"] < end)]
    return df[["site", "date", "actual_da", "naive_prediction", "ensemble_prediction",
               "spike_probability", "spike_alert"]].reset_index(drop=True)


def main() -> int:
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="baseline")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2022-01-01")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None, help="at most this many anchor dates")
    ap.add_argument("--out-dir", default=os.environ.get("EXP_OUT", "results/experiments/onset_raw"))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--seeds", default=None, help="comma-separated model seeds; files get __s<seed>")
    args = ap.parse_args()
    pts = test_points(args.start, args.end)
    pts["anchor"] = pts["date"] - pd.Timedelta(days=H)
    groups = [(a, g[["site", "date", "actual_da"]].to_dict("records")) for a, g in pts.groupby("anchor")]
    if args.limit:
        groups = groups[:: max(1, len(groups) // args.limit)][: args.limit]
    n_workers = args.n_jobs if args.n_jobs > 0 else os.cpu_count()
    chunks = [groups[i::n_workers * 4] for i in range(n_workers * 4)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else [None]
    runs = [(n, s) for n in names for s in seeds]
    tag = f"{args.start[:4]}_{args.end[:4]}"
    with Parallel(n_jobs=args.n_jobs, backend="loky") as par:
        for name, seed in runs:
            t0 = time.time()
            v = dict(VARIANTS[name]) if seed is None else {**VARIANTS[name], "seed": seed}
            stem = name if seed is None else f"{name}__s{seed}"
            parts = par(delayed(_run_chunk)(c, v) for c in chunks if c)
            res = pd.DataFrame([r for part in parts for r in part])
            res = res.merge(pts.drop(columns=["anchor"]), on=["site", "date"], how="left")
            res = res.sort_values(["date", "site"]).reset_index(drop=True)
            res.to_parquet(out / f"{tag}__{stem}.parquet", index=False)
            n_pts = sum(len(g) for _, g in groups)
            print(f"{tag} {stem:26s} n={len(res)}/{n_pts} anchors={len(groups)} {time.time() - t0:5.0f}s",
                  flush=True)
            if args.verify and name == "baseline" and seed in (None, config.MODEL_SEED):
                d = np.abs(res["spike_prob"] - res["spike_probability"])
                print(f"  verify: max |harness - committed| = {d.max():.3e} over {len(res)} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

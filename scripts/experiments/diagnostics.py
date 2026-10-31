#!/usr/bin/env python3
"""Error diagnostics on the committed deterministic chronological predictions.

Reads results/chronological/*.parquet (validation and holdout) and writes CSVs to
results/experiments/diagnostics/. Pure rescoring: no model is trained.

    python scripts/experiments/diagnostics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from scoring import SPIKE, paired_delta_r2, r2, summarize, transition_mask  # noqa: E402

WINDOWS = {
    "validation": "results/chronological/chronological_regression_ensemble_20190101_20220101.parquet",
    "holdout": "results/chronological/chronological_regression_ensemble_20220101_20240101.parquet",
}
OUT = ROOT / "results" / "experiments" / "diagnostics"
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "site"]).reset_index(drop=True)
    df["prev_actual"] = df.groupby("site")["actual_da"].shift(1)
    df["trans"] = transition_mask(df)
    prev_hi = df["naive_prediction"] >= SPIKE  # last measurement available at the anchor
    hi = df["actual_da"] >= SPIKE
    df["regime"] = np.select(
        [~prev_hi & ~hi, ~prev_hi & hi, prev_hi & hi, prev_hi & ~hi],
        ["quiet", "onset", "sustained", "decline"], default="?",
    )
    df["season"] = df["date"].dt.month.map(SEASON)
    df["mag"] = pd.cut(df["actual_da"], [-np.inf, 5, 20, 40, np.inf], right=False,
                       labels=["[0,5)", "[5,20)", "[20,40)", "[40,inf)"])
    df["sq_err"] = (df["predicted_da"] - df["actual_da"]) ** 2
    return df


def group_table(df: pd.DataFrame, key: str) -> pd.DataFrame:
    sse = df["sq_err"].sum()
    rows = []
    for k, g in df.groupby(key, observed=True):
        rows.append({
            key: k, "n": len(g), "mean_actual": g.actual_da.mean(), "mean_pred": g.predicted_da.mean(),
            "mean_naive": g.naive_prediction.mean(),
            "r2": r2(g.actual_da, g.predicted_da) if len(g) > 4 else np.nan,
            "r2_naive": r2(g.actual_da, g.naive_prediction) if len(g) > 4 else np.nan,
            "mae": (g.predicted_da - g.actual_da).abs().mean(),
            "mae_naive": (g.naive_prediction - g.actual_da).abs().mean(),
            "bias": (g.predicted_da - g.actual_da).mean(), "sse_share": g.sq_err.sum() / sse,
        })
    return pd.DataFrame(rows)


def influence(df: pd.DataFrame) -> dict:
    y, p = df.actual_da.to_numpy(), df.predicted_da.to_numpy()
    err = (y - p) ** 2
    order = np.argsort(-err)
    out = {"r2": r2(y, p), "r2_naive": r2(y, df.naive_prediction)}
    for k in (1, 3, 5, 10):
        keep = np.ones(len(y), bool)
        keep[order[:k]] = False
        out[f"r2_drop_top{k}_sqerr"] = r2(y[keep], p[keep])
        out[f"sse_share_top{k}"] = err[order[:k]].sum() / err.sum()
    big = np.argsort(-y)[:5]
    keep = np.ones(len(y), bool)
    keep[big] = False
    out["r2_drop_top5_actual"] = r2(y[keep], p[keep])
    for site in df.site.unique():
        m = (df.site != site).to_numpy()
        out[f"r2_without_{site.replace(' ', '_')}"] = r2(y[m], p[m])
    out["r2_log1p"] = r2(np.log1p(y), np.log1p(p))
    out["r2_log1p_naive"] = r2(np.log1p(y), np.log1p(df.naive_prediction))
    slope, intercept = np.polyfit(p, y, 1)
    out["calib_slope_actual_on_pred"] = slope
    out["calib_intercept"] = intercept
    out["pred_sd_over_actual_sd"] = p.std() / y.std()
    return out


def components(df: pd.DataFrame) -> pd.DataFrame:
    y = df.actual_da
    rows = []
    for name, col in [("ensemble", "predicted_da"), ("rf", "predicted_da_rf"),
                      ("ridge", "predicted_da_linear"), ("naive", "naive_prediction")]:
        d, lo, hi = paired_delta_r2(y, df.predicted_da, df[col])
        rows.append({"model": name, **summarize(y, df[col], df.trans.to_numpy()),
                     "d_r2_vs_ens": d, "lo": lo, "hi": hi})
    hyb = df["spike_alert"].fillna(False).astype(bool).to_numpy()
    rows.append({"model": "hybrid_alert", "recall": hyb[y >= SPIKE].mean(),
                 "trans_recall": hyb[df.trans].mean(),
                 "precision": (y[hyb] >= SPIKE).mean()})
    return pd.DataFrame(rows)


def calibration_bins(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 2, 5, 10, 20, 40, np.inf]
    lab = pd.cut(df.predicted_da, bins, right=False)
    return df.groupby(lab, observed=True).agg(
        n=("actual_da", "size"), mean_pred=("predicted_da", "mean"), mean_actual=("actual_da", "mean"),
        median_actual=("actual_da", "median"), spike_rate=("actual_da", lambda s: (s >= SPIKE).mean()),
    ).reset_index().rename(columns={"predicted_da": "pred_bin"})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    infl_rows = []
    for name, path in WINDOWS.items():
        df = load(path)
        print(f"\n######## {name}  N={len(df)}  spikes={(df.actual_da >= SPIKE).sum()}  "
              f"transitions={df.trans.sum()}")
        for key in ("site", "season", "mag", "regime"):
            t = group_table(df, key)
            t.to_csv(OUT / f"{name}_by_{key}.csv", index=False)
            print(t.round(3).to_string(index=False))
        comp = components(df)
        comp.to_csv(OUT / f"{name}_components.csv", index=False)
        print(comp.round(3).to_string(index=False))
        cal = calibration_bins(df)
        cal.to_csv(OUT / f"{name}_calibration.csv", index=False)
        print(cal.round(2).to_string(index=False))
        infl_rows.append({"window": name, **influence(df)})
        tr = df[df.trans]
        print(f"transitions: mean actual {tr.actual_da.mean():.1f}, mean pred {tr.predicted_da.mean():.1f}, "
              f"mean naive {tr.naive_prediction.mean():.1f}, pred>=12 {np.mean(tr.predicted_da >= 12):.2f}")
    infl = pd.DataFrame(infl_rows).set_index("window").T
    infl.to_csv(OUT / "influence.csv")
    print(infl.round(3).to_string())


if __name__ == "__main__":
    main()

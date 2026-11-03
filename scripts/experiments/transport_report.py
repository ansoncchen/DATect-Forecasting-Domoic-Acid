#!/usr/bin/env python3
"""Score the transport-trigger round (docs/FORECAST_EXPERIMENTS.md §7) with its declared rule.

Each model's score is its mean probability over seeds 42-46; the noise floor compares the
baseline over seeds 42-46 with seeds 47-51. The threshold is the lowest score keeping
<= 10% of 2015-2021 quiet weeks alarmed (all sites; WA rows on their own for the
secondary test), applied unchanged to the sub-windows. CIs resample calendar months and
re-choose both thresholds in every resample.

    python scripts/experiments/transport_report.py --raw-dir <onset_study out-dir>
    python scripts/experiments/transport_report.py --raw-dir <dir> --holdout wind   # passing variants only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from onset_report import (BUDGET, OUT, auc, block_resamples, candidates,  # noqa: E402
                          threshold_for, transfer)
from onset_study import REGIONS  # noqa: E402

TAG = "2015_2024"
SEEDS = (42, 43, 44, 45, 46)
FLOOR_SEEDS = (47, 48, 49, 50, 51)
VARIANTS = ("wind", "mooring", "wind+mooring", "wind+interactions")
SELECTION, SUBS = ("2015", "2022"), {"2015–2018": ("2015", "2019"), "2019–2021": ("2019", "2022")}
HOLDOUT = ("2022", "2024")
WA = set(REGIONS["WA"])


def mean_score(raw: Path, name: str, seeds) -> pd.DataFrame:
    frames = [pd.read_parquet(raw / f"{TAG}__{name}__s{s}.parquet") for s in seeds]
    base = frames[0][["site", "date", "actual_da", "naive_prediction", "spike_probability"]].copy()
    probs = np.column_stack([f.set_index(["site", "date"]).loc[list(zip(base.site, base.date)), "spike_prob"]
                             for f in frames])
    base["score"] = probs.mean(axis=1)
    for s, col in zip(seeds, probs.T):
        base[f"s{s}"] = col
    return base.sort_values(["date", "site"]).reset_index(drop=True)


def caught_at(onset: np.ndarray, score: np.ndarray, t: float) -> int:
    return int((score[onset] >= t).sum())


def compare(df0: pd.DataFrame, df1: pd.DataFrame, window: tuple, rows_mask=None) -> dict:
    """Variant (df1.score) vs baseline (df0.score) on the selection rows; thresholds chosen
    on ``window`` for each model, sub-windows scored at those thresholds."""
    m = (df0["date"] >= window[0]) & (df0["date"] < window[1])
    if rows_mask is not None:
        m &= rows_mask
    w0, w1 = df0[m].reset_index(drop=True), df1[m].reset_index(drop=True)
    cand, onset = candidates(w0)
    c = np.flatnonzero(cand)
    oc = onset[c]
    s0, s1 = w0["score"].to_numpy()[c], w1["score"].to_numpy()[c]
    t0, t1 = threshold_for(oc, s0, BUDGET), threshold_for(oc, s1, BUDGET)
    out = {"n_cand": len(c), "n_onset": int(oc.sum()), "threshold_base": t0, "threshold": t1,
           "caught_base": caught_at(oc, s0, t0), "caught": caught_at(oc, s1, t1),
           "auc_base": auc(oc, s0), "auc": auc(oc, s1)}
    out["d_caught"] = out["caught"] - out["caught_base"]
    out["d_auc"] = out["auc"] - out["auc_base"]
    boots = block_resamples(w0["date"].iloc[c].reset_index(drop=True))
    dc, da = [], []
    for idx in boots:
        o = oc[idx]
        if 0 < o.sum() < len(o):
            dc.append(caught_at(o, s1[idx], threshold_for(o, s1[idx], BUDGET))
                      - caught_at(o, s0[idx], threshold_for(o, s0[idx], BUDGET)))
            da.append(auc(o, s1[idx]) - auc(o, s0[idx]))
    out["d_caught_lo"], out["d_caught_hi"] = np.percentile(dc, [2.5, 97.5])
    out["d_auc_lo"], out["d_auc_hi"] = np.percentile(da, [2.5, 97.5])
    dates = w0["date"].iloc[c].to_numpy()
    for name, (a, b) in SUBS.items():
        sub = (dates >= np.datetime64(a)) & (dates < np.datetime64(b))
        out[f"d_caught {name}"] = caught_at(oc[sub], s1[sub], t1) - caught_at(oc[sub], s0[sub], t0)
    # per-seed paired deltas at each seed's own threshold (information only)
    per = [caught_at(oc, w1[f"s{s}"].to_numpy()[c], threshold_for(oc, w1[f"s{s}"].to_numpy()[c], BUDGET))
           - caught_at(oc, w0[f"s{s}"].to_numpy()[c], threshold_for(oc, w0[f"s{s}"].to_numpy()[c], BUDGET))
           for s in SEEDS if f"s{s}" in w1 and f"s{s}" in w0]
    out["per_seed_d_caught"] = ",".join(f"{x:+d}" for x in per)
    return out


def decide(r: dict, floor: float) -> bool:
    return bool(r["d_caught"] > floor and r["d_caught_lo"] > 0
                and all(r[f"d_caught {k}"] >= 0 for k in SUBS) and r["d_auc"] >= 0)


def deployment(df: pd.DataFrame, t: float, window: tuple, rows_mask=None) -> dict:
    m = (df["date"] >= window[0]) & (df["date"] < window[1])
    if rows_mask is not None:
        m &= rows_mask
    w = df[m]
    return transfer(w, w["score"].to_numpy(float), t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--holdout", default=None, help="comma-separated variants that passed (scored once)")
    args = ap.parse_args()
    raw = Path(args.raw_dir)
    base = mean_score(raw, "baseline", SEEDS)
    d = np.abs(pd.read_parquet(raw / f"{TAG}__baseline__s42.parquet").eval("spike_prob - spike_probability"))
    print(f"baseline seed 42 vs committed: max |diff| = {d.max():.2e}")
    floor_base = mean_score(raw, "baseline", FLOOR_SEEDS)
    models = {v: mean_score(raw, v, SEEDS) for v in VARIANTS}
    wa_mask = base["site"].isin(WA)
    rows, deploy = [], []
    for scope, mask in (("all sites", None), ("WA", wa_mask)):
        fl = compare(base, floor_base, SELECTION, mask)
        floor = max(2, abs(fl["d_caught"]))
        rows.append({"scope": scope, "variant": "baseline seeds 47–51 (noise floor)", **fl, "floor": floor, "passes": None})
        for v, df in models.items():
            r = compare(base, df, SELECTION, mask)
            rows.append({"scope": scope, "variant": v, **r, "floor": floor, "passes": decide(r, floor)})
        for name, df in {"baseline": base, **models}.items():
            t = rows[-1]["threshold_base"] if name == "baseline" else [x for x in rows if x["scope"] == scope and x["variant"] == name][0]["threshold"]
            for wname, win in {"2015–2021": SELECTION, **SUBS}.items():
                deploy.append({"scope": scope, "variant": name, "window": wname, "threshold": t,
                               **deployment(df, t, win, mask)})
    tab = pd.DataFrame(rows)
    dep = pd.DataFrame(deploy)
    tab.to_csv(OUT / "transport_selection.csv", index=False, float_format="%.4f")
    dep.to_csv(OUT / "transport_deployment.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    cols = ["scope", "variant", "n_onset", "caught_base", "caught", "d_caught", "d_caught_lo", "d_caught_hi",
            "d_caught 2015–2018", "d_caught 2019–2021", "auc_base", "auc", "d_auc", "d_auc_lo", "d_auc_hi",
            "per_seed_d_caught", "floor", "passes"]
    print(tab[cols].round(3).to_string(index=False))
    print(dep.round(3).to_string(index=False))
    if args.holdout:
        hold = []
        for scope, mask in (("all sites", None), ("WA", wa_mask)):
            t0 = tab[(tab.scope == scope) & (tab.variant == VARIANTS[0])]["threshold_base"].iloc[0]
            for name in ["baseline"] + args.holdout.split(","):
                df = base if name == "baseline" else models[name]
                t = t0 if name == "baseline" else tab[(tab.scope == scope) & (tab.variant == name)]["threshold"].iloc[0]
                hold.append({"scope": scope, "variant": name, "threshold": t, **deployment(df, t, HOLDOUT, mask)})
        hold = pd.DataFrame(hold)
        hold.to_csv(OUT / "transport_holdout.csv", index=False, float_format="%.4f")
        print(hold.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

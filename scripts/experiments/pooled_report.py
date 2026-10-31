#!/usr/bin/env python3
"""Full table for the declared pooled comparison (docs/FORECAST_EXPERIMENTS.md §3b).

Every candidate is reported, plus the current engine and the baselines on the same
rows, and the declared selection rule is applied mechanically (validation only).

    python scripts/experiments/pooled_report.py --window val --raw-dir results/experiments/raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from scoring import (  # noqa: E402
    ALERT, SPIKE, onset_metrics, paired_delta_auc, paired_delta_r2, paired_delta_recall_at_far,
    r2_ci, summarize,
)
from report_tables import fmt_table  # noqa: E402

FAMILIES = ("linear", "rf", "xgb")
ORDER = {"linear": 0, "rf": 1, "xgb": 2}


def table(df: pd.DataFrame) -> pd.DataFrame:
    y = df["actual_da"].to_numpy(float)
    naive = df["naive_prediction"].to_numpy(float)
    ref_onset = naive  # ΔAUC is reported against the last-value rule
    cands = [("Last-value rule / naive persistence", naive, naive)]
    if df["spike_probability"].notna().all():
        cands.append(("Current engine (classifier / per-site ensemble)",
                      df["spike_probability"].to_numpy(float), df["committed_ensemble"].to_numpy(float)))
    for fs in ("F1", "F2"):
        for fam in FAMILIES:
            if f"p_{fs}_{fam}" in df:
                cands.append((f"{fs} {fam}", df[f"p_{fs}_{fam}"].to_numpy(float), df[f"lvl_{fs}_{fam}"].to_numpy(float)))
    rows = []
    for name, p, lvl in cands:
        om = onset_metrics(naive, y, p)
        d, lo, hi = paired_delta_auc(naive, y, ref_onset, p)
        lvl_om = onset_metrics(naive, y, lvl)
        r2, r2lo, r2hi = r2_ci(y, lvl)
        s = summarize(y, lvl, alert=ALERT)
        rows.append({
            "candidate": name, "onset_auc": om["onset_auc"], "d_auc_vs_lastvalue": d, "d_lo": lo, "d_hi": hi,
            "recall_far10": om["onset_recall_far10"], "recall_far20": om["onset_recall_far20"],
            "gain_vs_lastvalue_far20": om["onset_vs_lastvalue_far20"],
            "level_r2": r2, "level_r2_lo": r2lo, "level_r2_hi": r2hi, "level_mae": s["mae"], "level_f2": s["f2"],
            "level_onset_auc": lvl_om["onset_auc"], "level_recall_far20": lvl_om["onset_recall_far20"],
            "n_onset": om["n_onset"], "n_cand": om["n_cand"],
        })
    return pd.DataFrame(rows)


def pairwise(df: pd.DataFrame, fs: str) -> pd.DataFrame:
    y = df["actual_da"].to_numpy(float)
    naive = df["naive_prediction"].to_numpy(float)
    out = []
    for a in FAMILIES:
        for b in FAMILIES:
            if ORDER[b] > ORDER[a]:
                d, lo, hi = paired_delta_auc(naive, y, df[f"p_{fs}_{a}"], df[f"p_{fs}_{b}"])
                out.append({"feature_set": fs, "complex": b, "simple": a, "d_auc": d, "lo": lo, "hi": hi})
    for fam in FAMILIES:
        d, lo, hi = paired_delta_auc(naive, y, df[f"p_F1_{fam}"], df[f"p_F2_{fam}"])
        out.append({"feature_set": "F2 vs F1", "complex": fam, "simple": fam, "d_auc": d, "lo": lo, "hi": hi})
    return pd.DataFrame(out)


def select(tab: pd.DataFrame, pw: pd.DataFrame) -> str:
    """Declared rule: best F1 onset AUC unless its CI vs a simpler family includes 0;
    F2 only if it beats F1 for that family with a CI excluding 0."""
    f1 = tab[tab["candidate"].str.startswith("F1 ")].copy()
    f1["fam"] = f1["candidate"].str.split().str[1]
    best = f1.sort_values("onset_auc", ascending=False)["fam"].iloc[0]
    for simple in sorted(FAMILIES, key=ORDER.get):
        if ORDER[simple] >= ORDER[best]:
            break
        r = pw[(pw.feature_set == "F1") & (pw.complex == best) & (pw.simple == simple)].iloc[0]
        if r["lo"] <= 0:
            best = simple
            break
    r = pw[(pw.feature_set == "F2 vs F1") & (pw.complex == best)].iloc[0]
    fs = "F2" if r["lo"] > 0 else "F1"
    return f"{fs} {best}"


def markdown(tab: pd.DataFrame) -> str:
    t = pd.DataFrame({
        "Candidate": tab["candidate"],
        "Onset AUC": tab["onset_auc"].map("{:.3f}".format),
        "ΔAUC vs last-value [95% CI]": [f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for d, lo, hi in
                                        zip(tab.d_auc_vs_lastvalue, tab.d_lo, tab.d_hi)],
        "Onset recall @10% / @20% FA": [f"{a:.2f} / {b:.2f}" for a, b in zip(tab.recall_far10, tab.recall_far20)],
        "Gain vs last-value @20%": tab["gain_vs_lastvalue_far20"].map("{:+.1f}".format),
        "Level R² [95% CI]": [f"{r:.3f} [{lo:.3f}, {hi:.3f}]" for r, lo, hi in
                              zip(tab.level_r2, tab.level_r2_lo, tab.level_r2_hi)],
        "MAE": tab["level_mae"].map("{:.2f}".format),
        "F2 (≥12)": tab["level_f2"].map("{:.3f}".format),
        "Level as onset score: AUC": tab["level_onset_auc"].map("{:.3f}".format),
    })
    return fmt_table(t)


def ablation(raw_dir: Path, window: str, fs: str, fam: str) -> pd.DataFrame:
    """Drop-one-group diagnostic vs the full feature set (same family, same rows)."""
    full = pd.read_parquet(raw_dir / f"{window}__pooled_study.parquet")
    rows = []
    for path in sorted(raw_dir.glob(f"{window}__pooled_drop-*.parquet")):
        group = path.stem.split("drop-", 1)[1]
        d = pd.read_parquet(path)
        if f"p_{fs}_{fam}" not in d:
            continue
        m = full.merge(d[["site", "date", f"p_{fs}_{fam}", f"lvl_{fs}_{fam}"]], on=["site", "date"],
                       suffixes=("", "_drop")).sort_values(["date", "site"])
        y, naive = m["actual_da"].to_numpy(float), m["naive_prediction"].to_numpy(float)
        a, b = m[f"p_{fs}_{fam}"], m[f"p_{fs}_{fam}_drop"]
        da = paired_delta_auc(naive, y, a, b)
        dr = paired_delta_recall_at_far(naive, y, a, b, 0.20)
        d2 = paired_delta_r2(y, m[f"lvl_{fs}_{fam}"], m[f"lvl_{fs}_{fam}_drop"])
        rows.append({"dropped": group, "d_onset_auc": da[0], "auc_lo": da[1], "auc_hi": da[2],
                     "d_recall_far20": dr[0], "rec_lo": dr[1], "rec_hi": dr[2],
                     "d_level_r2": d2[0], "r2_lo": d2[1], "r2_hi": d2[2]})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="val")
    ap.add_argument("--raw-dir", default="results/experiments/raw")
    ap.add_argument("--ablation", default=None, help="'<feature_set>:<family>' to report the drop-one-group table")
    args = ap.parse_args()
    if args.ablation:
        fs, fam = args.ablation.split(":")
        ab = ablation(Path(args.raw_dir), args.window, fs, fam)
        ab.to_csv(ROOT / "results" / "experiments" / f"pooled_ablation_{args.window}_{fs}_{fam}.csv",
                  index=False, float_format="%.4f")
        t = pd.DataFrame({
            "Group dropped": ab["dropped"],
            "Δ onset AUC [95% CI]": [f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for d, lo, hi in zip(ab.d_onset_auc, ab.auc_lo, ab.auc_hi)],
            "Δ onset recall @20% FA [95% CI]": [f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for d, lo, hi in zip(ab.d_recall_far20, ab.rec_lo, ab.rec_hi)],
            "Δ level R² [95% CI]": [f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for d, lo, hi in zip(ab.d_level_r2, ab.r2_lo, ab.r2_hi)],
        })
        print(fmt_table(t))
        return 0
    df = pd.read_parquet(Path(args.raw_dir) / f"{args.window}__pooled_study.parquet")
    df = df.sort_values(["date", "site"]).reset_index(drop=True)
    tab = table(df)
    pw = pairwise(df, "F1")
    pw = pd.concat([pw, pairwise(df, "F2").query("feature_set == 'F2'")], ignore_index=True)
    out = ROOT / "results" / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / f"pooled_{args.window}.csv", index=False, float_format="%.4f")
    pw.to_csv(out / f"pooled_{args.window}_pairwise.csv", index=False, float_format="%.4f")
    n_on = int(tab["n_onset"].iloc[0])
    print(f"window={args.window} rows={len(df)} candidates(last<20)={int(tab['n_cand'].iloc[0])} onsets={n_on}")
    print(markdown(tab))
    print()
    print(pw.round(3).to_string(index=False))
    if args.window == "val":
        print("\nDeclared rule selects:", select(tab, pw))
    return 0


if __name__ == "__main__":
    sys.exit(main())

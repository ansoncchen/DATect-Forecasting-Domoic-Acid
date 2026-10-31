#!/usr/bin/env python3
"""Print markdown tables from results/experiments/summary_<window>.csv for the write-up.

    python scripts/experiments/report_tables.py --window val [--match asof,clip]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def fmt_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def experiment_table(s: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Experiment": s["experiment"],
        "R²": s["r2"].map("{:.3f}".format),
        "ΔR² [95% CI]": [f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for d, lo, hi in zip(s.d_r2, s.d_r2_lo, s.d_r2_hi)],
        "MAE (Δ)": [f"{m:.2f} ({dm:+.2f})" for m, dm in zip(s.mae, s.d_mae)],
        "F2": s["f2"].map("{:.3f}".format),
        "Trans. recall": s["trans_recall"].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}"),
        "Verdict": s["verdict"],
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="val")
    ap.add_argument("--match", default="", help="comma-separated substrings to keep")
    args = ap.parse_args()
    s = pd.read_csv(ROOT / "results" / "experiments" / f"summary_{args.window}.csv")
    if args.match:
        keys = args.match.split(",")
        s = s[s["experiment"].apply(lambda e: any(k in e for k in keys))]
    print(fmt_table(experiment_table(s)))
    hyb = ROOT / "results" / "experiments" / f"summary_{args.window}_hybrid.csv"
    if hyb.exists() and not args.match:
        h = pd.read_csv(hyb)
        h = h.assign(**{c: h[c].map("{:.3f}".format) for c in ("recall", "precision", "f2", "trans_recall", "auc")})
        print()
        print(fmt_table(h[["experiment", "recall", "precision", "f2", "trans_recall", "auc", "n_alerts"]]))


if __name__ == "__main__":
    main()

"""
Print a concise numerical summary of all scores in outputs/scores/.

Reports:
  - Per-method mean/std score by region
  - Pearson correlation between method scores within each region (cross-method agreement)
  - Top-3 most-anomalous dates per method per region
  - E4 forecastability via compute_forecastability with bootstrap CI

This is a non-graphical companion to scripts/05_evaluate.py — useful for
spot-checking that the pipeline produced sensible numbers before opening figures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.evaluate import compute_forecastability
from src.regions import REGIONS


def main():
    scores_dir = config.SCORES_DIR
    files = [f for f in scores_dir.glob("*.parquet") if f.name != "all_scores.parquet"]
    if not files:
        print(f"No parquet files in {scores_dir}"); sys.exit(1)

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"skip {f}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["aggregation"] == "mean"]

    methods = sorted(df["method"].unique())
    regions = [r.name for r in REGIONS]
    print(f"\nLoaded {len(df)} rows  |  methods: {methods}  |  regions: {len(regions)}\n")

    # 1. Mean ± std per method per region
    print("=" * 92)
    print(f"{'Method':<40} {'Region':<35} {'Mean':>8} {'Std':>8}")
    print("-" * 92)
    for region in regions:
        for method in methods:
            s = df[(df["method"] == method) & (df["region"] == region)]["score"]
            if len(s):
                print(f"{method:<40} {region:<35} {s.mean():>8.3f} {s.std():>8.3f}")
        print()

    # 2. Cross-method correlations within Overall region
    overall = REGIONS[0].name
    sub = df[df["region"] == overall].pivot_table(
        index="date", columns="method", values="score", aggfunc="first"
    )
    print("=" * 92)
    print(f"Pearson correlation matrix (Overall region only)")
    corr = sub.corr().round(3)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(corr)
    print()

    # 3. Top-3 most-anomalous dates per method, Overall region
    print("=" * 92)
    print("Top-3 most-anomalous dates per method (Overall region)")
    for method in methods:
        s = df[(df["method"] == method) & (df["region"] == overall)].sort_values("score", ascending=False)
        top3 = s.head(3)
        dates_str = ", ".join(f"{r.date.date()}({r.score:.2f})" for r in top3.itertuples())
        print(f"  {method:<40} {dates_str}")
    print()

    # 4. Forecastability (R² + bootstrap CI) per region
    print("=" * 92)
    print("E4: One-step-ahead forecastability (R², CI of AE Δ vs matched PCA)")
    for region in regions:
        fore = compute_forecastability(df, region, n_bootstrap=500)
        if fore.empty:
            continue
        print(f"\n  {region}:")
        for _, row in fore.sort_values("r2", ascending=False).iterrows():
            ci = (f"  CIΔ=[{row['ci_low_vs_baseline']:+.3f}, {row['ci_high_vs_baseline']:+.3f}]"
                  f" vs {row['baseline_method']}"
                  if pd.notna(row["ci_low_vs_baseline"]) else "")
            print(f"    {row['method']:<40} R²={row['r2']:+.4f}{ci}")


if __name__ == "__main__":
    main()

"""
Produce E1–E5 + sanity figures from scores parquet files.

Usage:
    python scripts/05_evaluate.py
    python scripts/05_evaluate.py --scores-dir outputs/scores --out-dir outputs/figures
    python scripts/05_evaluate.py --aggregation mean
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.evaluate import (
    compute_forecastability, plot_seasonal_cycle, plot_event_timeseries,
    plot_forecastability, plot_channel_ablation, plot_yearly_drift,
)
from src.regions import REGIONS


def _slug(s: str) -> str:
    return s.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", default=str(config.SCORES_DIR))
    parser.add_argument("--out-dir", default=str(config.FIGURES_DIR))
    parser.add_argument("--aggregation", default="mean")
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = [f for f in scores_dir.glob("*.parquet") if f.name != "all_scores.parquet"]
    if not parquet_files:
        print(f"ERROR: no per-method parquet files in {scores_dir}")
        sys.exit(1)

    dfs = []
    for f in parquet_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  skip {f.name}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["aggregation"] == args.aggregation]

    methods = sorted(df["method"].unique())
    region_names = [r.name for r in REGIONS]
    print(f"Loaded {len(df)} rows  |  methods={methods}\n  regions={region_names}")

    # E1
    print("\nE1: Seasonal cycle…")
    for region in region_names:
        fig = plot_seasonal_cycle(df, region)
        fig.savefig(out_dir / f"E1_seasonal_{_slug(region)}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # E2
    print("E2: Event time series…")
    for region in region_names:
        fig = plot_event_timeseries(df, region)
        fig.savefig(out_dir / f"E2_events_{_slug(region)}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # E4
    print("E4: Forecastability with bootstrap CI…")
    fore_dfs = []
    for region in region_names:
        fore = compute_forecastability(df, region)
        if not fore.empty:
            fore_dfs.append(fore)
            print(f"  {region}:")
            for _, row in fore.sort_values("r2", ascending=False).iterrows():
                ci = (f"  CIΔ=[{row['ci_low_vs_baseline']:+.3f}, {row['ci_high_vs_baseline']:+.3f}]"
                      f" vs {row['baseline_method']}"
                      if pd.notna(row["ci_low_vs_baseline"]) else "")
                print(f"    {row['method']:40s}  R²={row['r2']:+.4f}{ci}")
    if fore_dfs:
        all_fore = pd.concat(fore_dfs, ignore_index=True)
        all_fore.to_parquet(out_dir / "E4_forecastability.parquet", index=False)
        fig = plot_forecastability(all_fore)
        fig.savefig(out_dir / "E4_forecastability.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # E5
        print("E5: Channel ablation…")
        for region in region_names:
            fig = plot_channel_ablation(all_fore, region)
            fig.savefig(out_dir / f"E5_ablation_{_slug(region)}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Sanity
    print("Sanity: yearly drift…")
    for region in region_names:
        fig = plot_yearly_drift(df, region)
        fig.savefig(out_dir / f"sanity_drift_{_slug(region)}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nAll figures → {out_dir}")


if __name__ == "__main__":
    main()

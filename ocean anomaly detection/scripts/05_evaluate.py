"""
Produce all evaluation figures (E1–E5 + sanity checks) from scores parquet files.

Usage:
    python scripts/05_evaluate.py
    python scripts/05_evaluate.py --scores-dir outputs/scores --out-dir outputs/figures
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.evaluate import (
    plot_seasonal_cycle,
    plot_event_timeseries,
    plot_forecastability,
    plot_channel_ablation,
    plot_yearly_drift,
    compute_forecastability,
)
from src.regions import REGIONS


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation figures")
    parser.add_argument("--scores-dir", default=str(config.SCORES_DIR))
    parser.add_argument("--out-dir", default=str(config.FIGURES_DIR))
    parser.add_argument("--aggregation", default="mean")
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all parquet files into one DataFrame
    parquet_files = list(scores_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"ERROR: no parquet files in {scores_dir}")
        print("Run scripts/04_run_inference.py first.")
        sys.exit(1)

    dfs = []
    for f in parquet_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  WARNING: could not read {f}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df_agg = df[df["aggregation"] == args.aggregation]

    methods = sorted(df_agg["method"].unique())
    region_names = [r.name for r in REGIONS]

    print(f"Loaded {len(df_agg)} rows, {len(methods)} methods, {len(region_names)} regions")
    print(f"Methods: {methods}")

    # -----------------------------------------------------------------------
    # E1 — Seasonal cycle
    # -----------------------------------------------------------------------
    print("\nE1: Seasonal cycle plots…")
    for region in region_names:
        fig = plot_seasonal_cycle(df_agg, region)
        path = out_dir / f"E1_seasonal_{region.replace(' ', '_').replace('/', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        fig.clf()
        import matplotlib.pyplot as plt; plt.close(fig)
        print(f"  {path}")

    # -----------------------------------------------------------------------
    # E2 — Event time series
    # -----------------------------------------------------------------------
    print("\nE2: Event time series…")
    for region in region_names:
        fig = plot_event_timeseries(df_agg, region)
        path = out_dir / f"E2_events_{region.replace(' ', '_').replace('/', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt; plt.close(fig)
        print(f"  {path}")

    # -----------------------------------------------------------------------
    # E4 — Forecastability (with bootstrap CI)
    # -----------------------------------------------------------------------
    print("\nE4: Forecastability (bootstrap R² CI)…")
    fore_dfs = []
    for region in region_names:
        fore = compute_forecastability(df_agg, region)
        if not fore.empty:
            fore_dfs.append(fore)
            print(f"  {region}:")
            for _, row in fore.sort_values("r2", ascending=False).iterrows():
                ci_str = (f"  CI_Δ=[{row['ci_low_vs_best_pca']:.3f}, {row['ci_high_vs_best_pca']:.3f}]"
                          if not pd.isna(row.get("ci_low_vs_best_pca", float("nan"))) else "")
                print(f"    {row['method']:40s}  R²={row['r2']:.4f}{ci_str}")

    if fore_dfs:
        all_fore = pd.concat(fore_dfs, ignore_index=True)
        all_fore.to_parquet(out_dir / "E4_forecastability.parquet", index=False)

        fig = plot_forecastability(all_fore)
        path = out_dir / "E4_forecastability.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt; plt.close(fig)
        print(f"  {path}")

        # -----------------------------------------------------------------------
        # E5 — Channel ablation
        # -----------------------------------------------------------------------
        print("\nE5: Channel ablation…")
        for region in region_names:
            fig = plot_channel_ablation(all_fore, region)
            path = out_dir / f"E5_ablation_{region.replace(' ', '_').replace('/', '_')}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt; plt.close(fig)
            print(f"  {path}")

    # -----------------------------------------------------------------------
    # Sanity: yearly drift
    # -----------------------------------------------------------------------
    print("\nSanity: Yearly drift…")
    for region in region_names:
        fig = plot_yearly_drift(df_agg, region)
        path = out_dir / f"sanity_drift_{region.replace(' ', '_').replace('/', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt; plt.close(fig)
        print(f"  {path}")

    print(f"\nAll figures written to {out_dir}")


if __name__ == "__main__":
    main()

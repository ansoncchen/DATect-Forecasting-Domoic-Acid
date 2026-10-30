"""
Sanity §8: cloud-coverage confound check.

For each (region, method) pair, compute Pearson r between OAD anomaly score
and the per-frame valid-pixel fraction inside that region. A high |r| means
the AE may be tracking cloud cover instead of actual ocean-state anomaly.

Outputs:
    outputs/figures/sanity_cloud_confound.parquet  — per (region, method) r + n
    outputs/figures/sanity_cloud_confound.png      — heatmap

Usage:
    python scripts/07_cloud_confound.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.regions import REGIONS


def compute_region_valid_fractions(cube_path: Path) -> pd.DataFrame:
    """For each (date, region), fraction of in-region pixels that are valid (4-channel ∩)."""
    ds = xr.open_zarr(cube_path)
    mask = ds["mask"]  # (time, lat, lon), bool — True = valid
    lat = ds.lat.values
    lon = ds.lon.values
    dates = pd.to_datetime(ds.time.values)

    rows = []
    for region in REGIONS:
        lat_in = (lat >= region.lat_min) & (lat <= region.lat_max)
        lon_in = (lon >= region.lon_min) & (lon <= region.lon_max)
        # rectangular bbox mask
        region_pixels = int(lat_in.sum() * lon_in.sum())
        if region_pixels == 0:
            print(f"  WARN: 0 pixels in {region.name}")
            continue
        # sum valid pixels per time step inside this region
        sub = mask.isel(lat=lat_in, lon=lon_in)
        valid_per_t = sub.sum(dim=["lat", "lon"]).values.astype(np.float64)
        frac = valid_per_t / region_pixels
        for d, f in zip(dates, frac):
            rows.append({"date": d, "region": region.name, "valid_frac": float(f)})
        print(f"  {region.name:40s}  pixels={region_pixels:6d}  "
              f"frac mean={frac.mean():.3f}  min={frac.min():.3f}  max={frac.max():.3f}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube", default=str(config.CUBE_PATH))
    parser.add_argument("--scores-dir", default=str(config.SCORES_DIR))
    parser.add_argument("--out-dir", default=str(config.FIGURES_DIR))
    parser.add_argument("--aggregation", default="mean")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing valid-pixel fraction per (date, region) from {args.cube} ...")
    valid_df = compute_region_valid_fractions(Path(args.cube))
    valid_df["date"] = pd.to_datetime(valid_df["date"])

    all_scores = Path(args.scores_dir) / "all_scores.parquet"
    print(f"\nLoading {all_scores} ...")
    df = pd.read_parquet(all_scores)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["aggregation"] == args.aggregation]
    print(f"  {len(df):,} rows, {df['method'].nunique()} methods, {df['region'].nunique()} regions")

    print("\nMerging scores with valid_frac and computing per-(region,method) Pearson r ...")
    merged = df.merge(valid_df, on=["date", "region"], how="inner")

    rows = []
    for (region, method), g in merged.groupby(["region", "method"]):
        g = g.dropna(subset=["score", "valid_frac"])
        if len(g) < 30:
            continue
        r = float(g[["score", "valid_frac"]].corr().iloc[0, 1])
        rows.append({"region": region, "method": method, "pearson_r": r, "n": len(g)})

    res = pd.DataFrame(rows).sort_values(["region", "pearson_r"], key=lambda s: s.abs() if s.name == "pearson_r" else s, ascending=[True, False])
    out_parquet = out_dir / "sanity_cloud_confound.parquet"
    res.to_parquet(out_parquet, index=False)
    print(f"\nWrote {out_parquet} ({len(res)} rows)")

    # Print summary: worst confound per region
    print("\n" + "=" * 80)
    print("Largest |r| per region (top 3 per region):")
    for region, g in res.groupby("region"):
        g_sorted = g.assign(abs_r=g["pearson_r"].abs()).sort_values("abs_r", ascending=False).head(3)
        print(f"\n  {region}:")
        for _, row in g_sorted.iterrows():
            flag = "  !!!" if abs(row["pearson_r"]) > 0.5 else ""
            print(f"    {row['method']:42s}  r={row['pearson_r']:+.3f}  n={row['n']:5d}{flag}")

    # Heatmap
    pivot = res.pivot(index="method", columns="region", values="pearson_r")
    methods_sorted = pivot.abs().mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[methods_sorted]

    fig, ax = plt.subplots(figsize=(8, max(6, 0.25 * len(methods_sorted))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-0.7, vmax=0.7)
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index, fontsize=6)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=5, color="white" if abs(v) > 0.4 else "black")
    fig.colorbar(im, ax=ax, label="Pearson r (score vs valid-pixel fraction)")
    ax.set_title("Sanity §8: cloud-coverage confound  —  |r|>0.5 ⇒ score may track clouds")
    fig.tight_layout()
    png = out_dir / "sanity_cloud_confound.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {png}")

    # Overall verdict
    worst = res["pearson_r"].abs().max()
    print("\n" + "=" * 80)
    print(f"VERDICT: max |r| across all (region, method) = {worst:.3f}")
    if worst > 0.5:
        print("  ⚠️  Some methods strongly track cloud cover. Restrict OAD use to high-valid-coverage dates,")
        print("      or pick a method with small |r| in your target region.")
    else:
        print("  ✅  All methods are weakly correlated with cloud cover (|r|<0.5).")
        print("      OAD scores can be used as-is without cloud-fraction conditioning.")


if __name__ == "__main__":
    main()

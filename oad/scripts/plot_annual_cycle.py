"""
Plot the anomaly score annual cycle stacked across years — answers
"do we see ocean anomalies pop up and fade every year?"

For each method × region:
  x-axis = day of year (1-365)
  y-axis = anomaly score
  one line per year, viridis color (early years dark, recent years bright)

Run on Hyak then scp the figure back.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

df = pd.read_parquet(config.SCORES_DIR / "all_scores.parquet")
df["date"] = pd.to_datetime(df["date"])
df["year"] = df.date.dt.year
df["doy"] = df.date.dt.dayofyear

# Pick representative methods
methods = ["AE_2d_l32", "AE_3d_l32_t4", "B3_pca_k32", "B2_multivar_zscore"]
methods = [m for m in methods if m in df.method.unique()]
regions = df.region.unique()
out_dir = config.FIGURES_DIR / "annual_cycle"
out_dir.mkdir(parents=True, exist_ok=True)

for region in regions:
    fig, axes = plt.subplots(len(methods), 1, figsize=(14, 3.0*len(methods)), sharex=True)
    if len(methods) == 1:
        axes = [axes]
    sub_r = df[df.region == region]
    years = sorted(sub_r.year.unique())
    colors = cm.viridis(np.linspace(0.1, 0.95, len(years)))

    for ax, method in zip(axes, methods):
        sub = sub_r[sub_r.method == method].copy()
        for i, year in enumerate(years):
            yr = sub[sub.year == year].sort_values("doy")
            if yr.empty:
                continue
            ax.plot(yr["doy"], yr["score"], color=colors[i],
                    alpha=0.55, lw=0.9, label=str(year) if i % 3 == 0 else None)
        # Mean cycle across all years (median smoothing)
        agg = sub.groupby("doy")["score"].median()
        ax.plot(agg.index, agg.values, color="black", lw=2.0, alpha=0.85,
                label=f"22-year median")
        ax.set_title(f"{method}", fontsize=10)
        ax.set_ylabel("Anomaly score")
        ax.legend(fontsize=7, loc="upper left", ncol=4, frameon=False)
        ax.grid(True, alpha=0.3)
        # Annotate canonical PNW seasons
        ax.axvspan(60, 152, color="lightgreen", alpha=0.10)   # Mar-May spring bloom
        ax.axvspan(152, 244, color="orange", alpha=0.10)      # Jun-Aug summer upwelling

    axes[-1].set_xlabel("Day of year")
    axes[-1].set_xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365])
    axes[-1].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec",""])
    safe = region.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    fig.suptitle(f"Annual cycle of ocean anomaly — {region}\n"
                 f"(green = Mar-May spring bloom, orange = Jun-Aug summer upwelling; black = 22-yr median)",
                 fontsize=10)
    fig.tight_layout()
    path = out_dir / f"annual_cycle_{safe}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# Also one combined plot — per-year time series for AE_3d only (cleanest event view)
fig, axes = plt.subplots(len(regions), 1, figsize=(14, 2.5*len(regions)), sharex=True)
if len(regions) == 1:
    axes = [axes]
for ax, region in zip(axes, regions):
    sub = df[(df.region == region) & (df.method == "AE_3d_l32_t4")].sort_values("date")
    ax.plot(sub.date, sub.score, color="#264653", lw=0.6, alpha=0.9)
    ax.set_title(f"AE_3d_l32_t4 — {region}", fontsize=9)
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    # Annotate known events
    ax.axvspan(pd.Timestamp("2014-09-01"), pd.Timestamp("2016-04-01"),
               alpha=0.18, color="#f4a261", label="MHW 2014-2016")
    ax.axvspan(pd.Timestamp("2015-05-01"), pd.Timestamp("2015-09-01"),
               alpha=0.18, color="#e76f51", label="PN bloom 2015")
    ax.axvspan(pd.Timestamp("2019-05-01"), pd.Timestamp("2019-12-01"),
               alpha=0.18, color="#2a9d8f", label="MHW 2019")
    if ax == axes[0]:
        ax.legend(fontsize=7, loc="upper right", ncol=3, frameon=False)
axes[-1].set_xlabel("Date")
fig.suptitle("Full 22-year anomaly time series across all regions (AE_3d)", fontsize=11)
fig.tight_layout()
fig.savefig(out_dir / "fullseries_ae3d_allregions.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_dir / 'fullseries_ae3d_allregions.png'}")

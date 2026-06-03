"""Regenerate the mae070 paper figures from local data.

The original one-off generators for fig4 (climatology), fig5 (ESP scatter), and
fig9 (ESP time series) were not committed and are lost; the committed PNGs were
built from the mae050 checkpoint (and fig5 did not even match its own Table 2).
This script rebuilds them from the mae070 checkpoint, matching the corrected
tables exactly, and is committed so the figures are reproducible going forward.

Inputs (all local):
  - ocean anomaly detection/outputs/scores/ae_3d_l32_c4_t4_s42_mae070.parquet
  - /Users/ansonchen/Downloads/ChaBa ESP database.xlsx  (cELISA=pDA, SHA=Pn cells)

Outputs written into BOTH paper figure dirs (paper_oad/figures, paper_oad_cvpr/figures):
  - fig4_climatology_timeseries.png
  - fig5_esp_validation.png
  - fig9_oad_esp_timeseries.png  (+ _wide for CVPR)

Cloud-confound scatter (fig7) needs per-day valid-pixel fraction from the cube
(Hyak only) and is NOT regenerated here.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[2]
SCORES = REPO / "ocean anomaly detection/outputs/scores/ae_3d_l32_c4_t4_s42_mae070.parquet"
ESP_XLSX = Path("/Users/ansonchen/Downloads/ChaBa ESP database.xlsx")
OUT_DIRS = [REPO / "paper_oad/figures", REPO / "paper_oad_cvpr/figures"]

NAVY = "#1d3557"
PURPLE = "#5b2a86"
RED = "#c1121f"
ORANGE = "#e76f51"

ALONGSHORE = [
    "Olympic Coast (WA)",
    "SW Washington / Long Beach",
    "Central Oregon",
    "Southern OR / N CA",
]


def load_scores():
    sc = pd.read_parquet(SCORES)
    sc = sc[sc["aggregation"] == "mean"].copy()
    sc["date"] = pd.to_datetime(sc["date"])
    return sc


def load_esp():
    xl = pd.ExcelFile(ESP_XLSX)
    c = xl.parse("cELISA")
    c["date"] = pd.to_datetime(c["Date"])
    pda = c.groupby("date")["DA concentration (ng/L)"].mean().rename("pda").reset_index()
    s = xl.parse("SHA")
    s["date"] = pd.to_datetime(s["Date"])
    cell_cols = [x for x in s.columns if "(cells/L)" in x]
    s["pn"] = s[cell_cols].sum(axis=1, min_count=1)
    pn = s.groupby("date")["pn"].mean().reset_index()
    return pn, pda


def boot_ci(x, y, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n):
        b = rng.integers(0, len(x), len(x))
        if np.std(x[b]) > 0 and np.std(y[b]) > 0:
            rs.append(np.corrcoef(x[b], y[b])[0, 1])
    return np.percentile(rs, [2.5, 97.5])


def save(fig, name):
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- fig5 scatter
def fig5_esp_scatter(sc, pn, pda):
    panels = [
        ("Olympic Coast (WA)", pn, "pn", "Pn cell density, cells/L", "Olympic Coast (WA)"),
        ("SW Washington / Long Beach", pn, "pn", "Pn cell density, cells/L", "SW Washington / Long Beach"),
        ("Olympic Coast (WA)", pda, "pda", "pDA, ng/L", "Olympic Coast (WA)"),
        ("SW Washington / Long Beach", pda, "pda", "pDA, ng/L", "SW Washington / Long Beach"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6))
    for ax, (region, tdf, col, ylab, title) in zip(axes.ravel(), panels):
        m = sc[sc["region"] == region][["date", "score"]].merge(tdf, on="date").dropna()
        x = m["score"].values
        y = m[col].values
        r, p = pearsonr(x, y)
        lo, hi = boot_ci(x, y)
        color = NAVY if col == "pn" else ORANGE
        ax.scatter(x, np.log10(y + 1), s=18, alpha=0.6, color=color, edgecolor="none")
        b1, b0 = np.polyfit(x, np.log10(y + 1), 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, b0 + b1 * xs, color=RED, lw=1.6, label="OLS fit")
        ax.set_title(title, fontsize=9, weight="bold")
        ax.set_xlabel("OAD anomaly score (z-units)", fontsize=8)
        ax.set_ylabel(rf"$\log_{{10}}$({ylab} + 1)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.text(0.02, 0.98,
                f"r = {r:+.3f} [{lo:+.3f}, {hi:+.3f}]\np = {p:.2e},  N = {len(m)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.2,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle("OAD anomaly score (mae070) vs. in-situ ESP measurements at NEMO mooring (2016–2018 ChaBa)",
                 fontsize=10, weight="bold", y=1.00)
    fig.tight_layout()
    save(fig, "fig5_esp_validation.png")


# ----------------------------------------------------------- fig9 time series
def deployment_windows(dates, gap_days=21):
    dates = sorted(pd.to_datetime(pd.unique(dates)))
    dates = [pd.Timestamp(d) for d in dates]
    windows = []
    start = prev = dates[0]
    for d in dates[1:]:
        if (d - prev).days > gap_days:
            windows.append((start, prev))
            start = d
        prev = d
    windows.append((start, prev))
    return windows


def fig9_timeseries(sc, pn, pda, wide=False):
    windows = deployment_windows(pd.concat([pn["date"], pda["date"]]))
    n = len(windows)
    figsize = (3.0 * n, 5.2) if wide else (2.3 * n, 5.2)
    fig, axes = plt.subplots(2, n, figsize=figsize, sharex="col")
    if n == 1:
        axes = axes.reshape(2, 1)
    rows = [
        ("Olympic Coast (WA)", pn, "pn", "Pn cells/L", NAVY),
        ("SW Washington / Long Beach", pda, "pda", "pDA ng/L", ORANGE),
    ]
    for ri, (region, tdf, col, ylab, mcol) in enumerate(rows):
        s_reg = sc[sc["region"] == region][["date", "score"]].sort_values("date")
        for ci, (w0, w1) in enumerate(windows):
            ax = axes[ri, ci]
            pad = pd.Timedelta(days=14)
            sw = s_reg[(s_reg["date"] >= w0 - pad) & (s_reg["date"] <= w1 + pad)]
            ax.plot(sw["date"], sw["score"], color=PURPLE, lw=1.3, zorder=3)
            ax.axvspan(w0, w1, color="#fff3b0", zorder=0)
            ax2 = ax.twinx()
            tw = tdf[(tdf["date"] >= w0 - pad) & (tdf["date"] <= w1 + pad)].dropna()
            ax2.scatter(tw["date"], tw[col], color=mcol, s=14, zorder=4)
            ax2.set_yscale("log")
            ax.tick_params(labelsize=6, axis="x", rotation=45)
            ax.tick_params(labelsize=6, axis="y")
            ax2.tick_params(labelsize=6)
            if ci == 0:
                ax.set_ylabel(f"OAD (z)\n{region.split(' (')[0]}", fontsize=7, color=PURPLE)
            if ci == n - 1:
                ax2.set_ylabel(ylab, fontsize=7, color=mcol)
            if ri == 0:
                ax.set_title(f"{w0.year}", fontsize=8, weight="bold")
    fig.suptitle("OAD score (mae070, purple) vs. in-water ESP measurements (markers) — 2016–2018 ChaBa deployments",
                 fontsize=9.5, weight="bold")
    fig.tight_layout()
    save(fig, "fig9_oad_esp_timeseries_wide.png" if wide else "fig9_oad_esp_timeseries.png")


# ------------------------------------------------------------ fig4 climatology
def fig4_climatology(sc):
    df = sc[sc["region"].isin(ALONGSHORE)].copy()
    df["ym"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby(["region", "ym"])["score"].mean().reset_index()
    fig, axes = plt.subplots(len(ALONGSHORE), 1, figsize=(9.5, 6.4), sharex=True)
    for ax, region in zip(axes, ALONGSHORE):
        g = monthly[monthly["region"] == region].sort_values("ym")
        # break the line across the 2009-2011 MODIS coverage gap
        gap0, gap1 = pd.Timestamp("2009-01-01"), pd.Timestamp("2011-12-31")
        gg = g.copy()
        gg.loc[(gg["ym"] >= gap0) & (gg["ym"] <= gap1), "score"] = np.nan
        ax.axvspan(gap0, gap1, color="0.9", zorder=0)
        ax.axvspan(pd.Timestamp("2014-01-01"), pd.Timestamp("2016-12-31"),
                   color="#fde0dd", zorder=0)
        ax.plot(gg["ym"], gg["score"], color=NAVY, lw=1.0)
        ax.set_ylabel(region.split(" (")[0], fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_title("22 years of OAD ocean-anomaly score (mae070, 2003–2024) — "
                      "the 2014–16 marine heatwave ('the Blob', shaded) is the record anomaly",
                      fontsize=10, weight="bold")
    axes[-1].set_xlabel("Year", fontsize=9)
    fig.tight_layout()
    save(fig, "fig4_climatology_timeseries.png")


# ------------------------------------------------------------ fig2 architecture
def fig2_architecture():
    from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(0.2, 6.55, "OAD — 3D Masked-Autoencoder Architecture",
            fontsize=15, weight="bold", color=PURPLE)
    ax.text(0.2, 6.05,
            "Unsupervised: reconstruct masked MODIS structure, then turn reconstruction error into a per-region ocean-anomaly score.",
            fontsize=8.5, color="0.35")

    def box(x, y, w, h, fc):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="none", zorder=2))

    def trap(x, yc, h_in, h_out, fc, label):
        pts = [(x, yc - h_in / 2), (x, yc + h_in / 2),
               (x + 2.2, yc + h_out / 2), (x + 2.2, yc - h_out / 2)]
        ax.add_patch(Polygon(pts, fc=fc, ec="none", zorder=2))
        ax.text(x + 1.1, yc, label, ha="center", va="center",
                color="white", fontsize=9, weight="bold")

    # input
    box(0.3, 3.0, 1.5, 1.5, "#2a9d8f")
    ax.text(1.05, 2.7, "Masked MODIS input\n4 ch × 4 days, 64×64 patch\n70% pixels hidden\nchl-a · Kd490 · nFLH · SST",
            ha="center", va="top", fontsize=7)
    # encoder (wide -> narrow)
    trap(2.4, 3.75, 2.4, 0.9, PURPLE, "ENCODER\n3× stride-2\nConv3D")
    # latent
    box(5.0, 3.45, 0.7, 0.6, "#e9c46a")
    ax.text(5.35, 3.15, "latent z\ndim 32", ha="center", va="top", fontsize=7.5)
    # decoder (narrow -> wide)
    trap(6.0, 3.75, 0.9, 2.4, PURPLE, "DECODER\n3× transposed\nConv3D")
    # reconstruction
    box(8.7, 3.0, 1.5, 1.5, "#457b9d")
    ax.text(9.45, 2.7, "Reconstruction", ha="center", va="top", fontsize=7.5)
    # masked MSE arrow (top)
    ax.add_patch(FancyArrowPatch((1.05, 4.85), (9.45, 4.85),
                 connectionstyle="arc3,rad=-0.18", arrowstyle="-|>",
                 mutation_scale=16, color=RED, lw=1.4, zorder=1))
    ax.text(5.2, 5.55, "Masked MSE — loss only on hidden ocean pixels",
            ha="center", fontsize=9, color=RED, weight="bold")
    # score derivation (bottom)
    ax.add_patch(FancyArrowPatch((9.45, 2.9), (9.45, 1.7), arrowstyle="-|>",
                 mutation_scale=14, color="0.4", lw=1.2))
    ax.text(10.3, 2.3, "anchor-frame error\n→ mean over region's\nocean pixels · z-score",
            ha="left", va="center", fontsize=6.8, color="0.3")
    box(4.6, 0.55, 3.2, 1.0, "#e76f51")
    ax.text(6.2, 1.05, "OAD anomaly score", ha="center", va="center",
            color="white", fontsize=11, weight="bold")
    ax.text(6.2, 0.72, "one scalar per region · per day", ha="center", va="center",
            color="white", fontsize=7.5)
    ax.add_patch(FancyArrowPatch((9.45, 1.7), (7.8, 1.05), arrowstyle="-|>",
                 mutation_scale=12, color="0.4", lw=1.0))
    ax.text(0.3, 0.15, "→ label-free evaluation:  forecastability · ESP validation · climate events  "
                       "(checkpoint: ae_3d_l32_c4_t4_s42_mae070)",
            fontsize=7.5, color="0.4")
    fig.tight_layout()
    save(fig, "fig2_architecture.png")


# --------------------------------------------------- fig6 seasonal cycle SW WA
def fig6_seasonal(sc):
    g = sc[sc["region"] == "SW Washington / Long Beach"].copy()
    g["doy"] = g["date"].dt.dayofyear
    by = g.groupby("doy")["score"]
    mean = by.mean()
    q25, q75 = by.quantile(0.25), by.quantile(0.75)
    lo, hi = by.min(), by.max()
    x = mean.index.values
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.fill_between(x, lo, hi, color=NAVY, alpha=0.10, label="full range")
    ax.fill_between(x, q25, q75, color=NAVY, alpha=0.25, label="inter-quartile range")
    ax.plot(x, mean.values, color=NAVY, lw=1.8, label="mean climatology")
    ax.set_xlabel("Day of year", fontsize=9)
    ax.set_ylabel("OAD anomaly score (z-units)", fontsize=9)
    ax.set_title("Seasonal cycle of the OAD score (mae070) — SW Washington, 2003–2024",
                 fontsize=10, weight="bold")
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "fig6_seasonal_cycle_sw_wa.png")


# ------------------------------------------------ fig8 event time series SW WA
def fig8_event(sc):
    g = sc[sc["region"] == "SW Washington / Long Beach"].copy().sort_values("date")
    g["ym"] = g["date"].dt.to_period("M").dt.to_timestamp()
    monthly = g.groupby("ym")["score"].mean()
    gap0, gap1 = pd.Timestamp("2009-01-01"), pd.Timestamp("2011-12-31")
    mm = monthly.copy()
    mm[(mm.index >= gap0) & (mm.index <= gap1)] = np.nan
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.axvspan(gap0, gap1, color="0.9", zorder=0, label="MODIS gap")
    ax.axvspan(pd.Timestamp("2014-06-01"), pd.Timestamp("2016-06-01"),
               color="#fde0dd", zorder=0, label="2014–16 'Blob'")
    ax.plot(mm.index, mm.values, color=NAVY, lw=1.0)
    for yr in (2015, 2019):
        ax.axvline(pd.Timestamp(f"{yr}-07-01"), color=RED, ls="--", lw=1.0, alpha=0.7)
        ax.text(pd.Timestamp(f"{yr}-07-01"), ax.get_ylim()[1], f" {yr} bloom",
                color=RED, fontsize=7, va="top")
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("OAD score (z-units)", fontsize=9)
    ax.set_title("Multi-year OAD score (mae070) with documented HAB events — SW Washington",
                 fontsize=10, weight="bold")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "fig8_event_detection_sw_wa.png")


def main():
    sc = load_scores()
    pn, pda = load_esp()
    fig2_architecture()
    fig6_seasonal(sc)
    fig8_event(sc)
    fig5_esp_scatter(sc, pn, pda)
    fig9_timeseries(sc, pn, pda, wide=False)
    fig9_timeseries(sc, pn, pda, wide=True)
    fig4_climatology(sc)


if __name__ == "__main__":
    main()

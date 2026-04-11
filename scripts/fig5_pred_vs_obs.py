"""
Generate publication-quality predicted vs observed time series figure
for the DATect paper (Figure 5).

Reads dev-set retrospective predictions (seed=42) for Copalis,
the highest-skill site (R^2 = 0.789).

Output: paper/figures/fig5_pred_vs_obs.png  (300 DPI, ~7 in wide)
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

# ── paths ────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
PARQUET = ROOT / "cache" / "retrospective" / "regression_ensemble.parquet"
OUT = ROOT / "paper" / "figures" / "fig5_pred_vs_obs.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── style ────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── data ─────────────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET)
site_name = "Copalis"
cop = df[df["site"] == site_name].copy()
cop["date"] = pd.to_datetime(cop["date"])
cop = cop.sort_values("date").reset_index(drop=True)

actual = cop["actual_da"].values
predicted = cop["predicted_da"].values

# Compute metrics
ss_res = np.sum((actual - predicted) ** 2)
ss_tot = np.sum((actual - np.mean(actual)) ** 2)
r2 = 1 - ss_res / ss_tot
mae = np.mean(np.abs(actual - predicted))
n = len(actual)

print(f"{site_name}: n={n}, R²={r2:.3f}, MAE={mae:.2f} µg/g")

# ── colours ──────────────────────────────────────────────────────────────
COL_ACTUAL = "#1a1a1a"       # near-black
COL_PRED = "#2166ac"         # steel blue
COL_THRESH = "#d6604d"       # muted red
COL_SCATTER = "#4393c3"      # lighter blue
COL_FILL = "#d1e5f0"         # pale blue fill between curves

# ── figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.0, 3.8))

# GridSpec: time series (wide) + scatter inset (narrow)
gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.35)

# ── panel A: time series ────────────────────────────────────────────────
ax = fig.add_subplot(gs[0])

# Light fill between actual and predicted
ax.fill_between(
    cop["date"], actual, predicted,
    alpha=0.15, color=COL_FILL, zorder=1,
)

# Predicted line (behind dots)
ax.plot(
    cop["date"], predicted,
    color=COL_PRED, linewidth=1.0, alpha=0.85,
    label="Ensemble prediction", zorder=2,
)

# Actual observations as dots connected by thin line
ax.plot(
    cop["date"], actual,
    color=COL_ACTUAL, linewidth=0.4, alpha=0.35, zorder=3,
)
ax.scatter(
    cop["date"], actual,
    s=14, color=COL_ACTUAL, edgecolors="none",
    label="Observed DA", zorder=4,
)

# Regulatory threshold
ax.axhline(
    20, color=COL_THRESH, linestyle="--", linewidth=0.9, alpha=0.7,
    label="Action threshold (20 \u00b5g/g)", zorder=1,
)

# Axes
ax.set_xlabel("Date")
ax.set_ylabel("Domoic acid (\u00b5g/g)")
ax.set_title(site_name, fontweight="bold", loc="left")
ax.set_ylim(bottom=-2)

# Date formatting
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_minor_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Subtle grid on y only
ax.yaxis.grid(True, linewidth=0.3, alpha=0.4, color="#cccccc")
ax.set_axisbelow(True)

ax.legend(loc="upper left", frameon=True, framealpha=0.9,
          edgecolor="#cccccc", fancybox=False)

# Panel label
ax.text(-0.08, 1.05, "(a)", transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top")

# ── panel B: 1:1 scatter ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])

ax2.scatter(
    actual, predicted,
    s=16, color=COL_SCATTER, edgecolors="white", linewidths=0.3,
    alpha=0.75, zorder=3,
)

# 1:1 line
lim_max = max(actual.max(), predicted.max()) * 1.05
ax2.plot([0, lim_max], [0, lim_max],
         color="#888888", linewidth=0.8, linestyle="--", zorder=2)

# Threshold lines
ax2.axhline(20, color=COL_THRESH, linewidth=0.6, linestyle=":", alpha=0.5)
ax2.axvline(20, color=COL_THRESH, linewidth=0.6, linestyle=":", alpha=0.5)

ax2.set_xlabel("Observed DA (\u00b5g/g)")
ax2.set_ylabel("Predicted DA (\u00b5g/g)")
ax2.set_xlim(-2, lim_max)
ax2.set_ylim(-2, lim_max)
ax2.set_aspect("equal", adjustable="box")
ax2.set_title("Predicted vs. observed", fontweight="bold", loc="left",
              fontsize=9)

# Annotate R² and MAE
ax2.text(
    0.05, 0.92,
    f"$R^2$ = {r2:.3f}\nMAE = {mae:.1f} \u00b5g/g\n$n$ = {n}",
    transform=ax2.transAxes, fontsize=8,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor="#cccccc", alpha=0.9),
)

# Panel label
ax2.text(-0.15, 1.05, "(b)", transform=ax2.transAxes,
         fontsize=11, fontweight="bold", va="top")

# Subtle grid
ax2.yaxis.grid(True, linewidth=0.3, alpha=0.4, color="#cccccc")
ax2.xaxis.grid(True, linewidth=0.3, alpha=0.4, color="#cccccc")
ax2.set_axisbelow(True)

# ── save ─────────────────────────────────────────────────────────────────
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
plt.close(fig)

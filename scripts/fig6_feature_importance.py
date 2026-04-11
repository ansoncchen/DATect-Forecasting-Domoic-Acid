"""Generate publication-quality feature importance bar chart (Figure 6)."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Load data and aggregate feature importances
# ---------------------------------------------------------------------------
DATA = Path(__file__).resolve().parents[1] / "cache/retrospective/regression_ensemble.parquet"
df = pd.read_parquet(DATA)

agg: dict[str, list[float]] = defaultdict(list)
for d in df["feature_importance"]:
    if d is None:
        continue
    for k, v in d.items():
        if v is not None:
            agg[k].append(v)

mean_imp = {k: np.mean(v) for k, v in agg.items()}

# ---------------------------------------------------------------------------
# 2. Human-readable labels
# ---------------------------------------------------------------------------
LABELS = {
    "last_observed_da_raw": "Last observed DA",
    "weeks_since_last_spike": "Weeks since spike",
    "da_raw_prev_obs_1": "Lag 1 (obs. order)",
    "da_raw_prev_obs_2": "Lag 2 (obs. order)",
    "da_raw_prev_obs_3": "Lag 3 (obs. order)",
    "da_raw_prev_obs_4": "Lag 4 (obs. order)",
    "da_raw_prev_obs_2_weeks_ago": "Lag 2 recency (weeks)",
    "da_raw_prev_obs_3_weeks_ago": "Lag 3 recency (weeks)",
    "da_raw_prev_obs_diff_1_2": "Lag trend (1\u20132)",
    "raw_obs_roll_max_4": "Rolling 4-wk max",
    "raw_obs_roll_max_8": "Rolling 8-wk max",
    "raw_obs_roll_max_12": "Rolling 12-wk max",
    "raw_obs_roll_mean_4": "Rolling 4-wk mean",
    "raw_obs_roll_std_4": "Rolling 4-wk std",
    "raw_obs_roll_std_8": "Rolling 8-wk std",
    "raw_obs_roll_std_12": "Rolling 12-wk std",
    "modis-sst": "SST",
    "sst-anom": "SST anomaly",
    "beuti": "BEUTI",
    "pdo": "PDO",
    "oni": "ONI",
    "discharge": "River discharge",
    "modis-flr": "FLH",
    "sin_day_of_year": "Sin(day of year)",
    "cos_day_of_year": "Cos(day of year)",
    "month": "Month",
    "pn_log": "log(PN)",
    "pn": "PN",
}

# ---------------------------------------------------------------------------
# 3. Category assignments and colors
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Persistence": [
        "last_observed_da_raw", "weeks_since_last_spike",
    ],
    "Observation-order lags": [
        "da_raw_prev_obs_1", "da_raw_prev_obs_2", "da_raw_prev_obs_3",
        "da_raw_prev_obs_4", "da_raw_prev_obs_2_weeks_ago",
        "da_raw_prev_obs_3_weeks_ago", "da_raw_prev_obs_diff_1_2",
    ],
    "Rolling statistics": [
        "raw_obs_roll_max_4", "raw_obs_roll_max_8", "raw_obs_roll_max_12",
        "raw_obs_roll_mean_4", "raw_obs_roll_std_4", "raw_obs_roll_std_8",
        "raw_obs_roll_std_12",
    ],
    "Environmental": [
        "modis-sst", "sst-anom", "beuti", "pdo", "oni", "discharge",
        "modis-flr",
    ],
    "Temporal": [
        "sin_day_of_year", "cos_day_of_year", "month",
    ],
    "Biological": [
        "pn_log", "pn",
    ],
}

COLORS = {
    "Persistence": "#1b4f72",       # dark blue
    "Observation-order lags": "#2e86c1",  # medium blue
    "Rolling statistics": "#85c1e9",      # light blue
    "Environmental": "#27ae60",     # green
    "Temporal": "#e67e22",          # orange
    "Biological": "#8e44ad",        # purple
}

feat_to_cat = {}
for cat, feats in CATEGORIES.items():
    for f in feats:
        feat_to_cat[f] = cat

# ---------------------------------------------------------------------------
# 4. Select top 15 and sort
# ---------------------------------------------------------------------------
top15 = sorted(mean_imp.items(), key=lambda x: x[1], reverse=True)[:15]
# Reverse so highest is at top of horizontal bar chart
top15.reverse()

features = [f for f, _ in top15]
values = [v for _, v in top15]
colors = [COLORS[feat_to_cat.get(f, "Environmental")] for f in features]
labels = [LABELS.get(f, f) for f in features]

# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.barh(range(len(features)), values, color=colors, edgecolor="white",
               linewidth=0.4, height=0.7)

ax.set_yticks(range(len(features)))
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("Mean feature importance (gain)", fontsize=10.5)
ax.set_title("Feature Importance: XGBoost + Random Forest Ensemble",
             fontsize=11.5, fontweight="bold", pad=12)

# Light grid on x only
ax.xaxis.grid(True, alpha=0.25, linewidth=0.5, color="#888888")
ax.yaxis.grid(False)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
legend_patches = [
    mpatches.Patch(color=COLORS[cat], label=cat)
    for cat in ["Persistence", "Observation-order lags", "Rolling statistics",
                "Environmental", "Temporal", "Biological"]
    # Only include categories that appear in the top 15
    if any(feat_to_cat.get(f) == cat for f in features)
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=8.5,
          framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout()

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
out = Path(__file__).resolve().parents[1] / "paper/figures/fig6_feature_importance.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved to {out}")
plt.close()

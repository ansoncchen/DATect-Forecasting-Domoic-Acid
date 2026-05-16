"""
Evaluation functions for E1–E5 + sanity checks (proposal §7–8).

All inputs are score DataFrames with columns:
    date, region, method, aggregation, score
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ---------------------------------------------------------------------------
# E1 — Seasonal cycle plausibility
# ---------------------------------------------------------------------------

def plot_seasonal_cycle(df: pd.DataFrame, region: str) -> plt.Figure:
    sub = df[df["region"] == region].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub["month"] = sub["date"].dt.month
    methods = sorted(sub["method"].unique())

    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        m = sub[sub["method"] == method]
        groups = [m[m["month"] == mo]["score"].dropna().values for mo in range(1, 13)]
        ax.boxplot(groups, tick_labels=range(1, 13), showfliers=False)
        ax.set_title(method, fontsize=8)
        ax.set_xlabel("Month")
        ax.set_ylabel("Anomaly score")
    fig.suptitle(f"E1 — Seasonal cycle: {region}", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# E2 — Known event reproduction
# ---------------------------------------------------------------------------

def plot_event_timeseries(df: pd.DataFrame, region: str) -> plt.Figure:
    sub = df[df["region"] == region].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    methods = sorted(sub["method"].unique())

    fig, axes = plt.subplots(len(methods), 1, figsize=(14, 3 * len(methods)), sharex=True)
    if len(methods) == 1:
        axes = [axes]

    colors = {"MHW 2014–2016": "#f4a261", "PN bloom 2015": "#e76f51", "MHW 2019": "#2a9d8f"}
    for ax, method in zip(axes, methods):
        m = sub[sub["method"] == method].sort_values("date")
        ax.plot(m["date"], m["score"], linewidth=0.8, color="#264653")
        ax.set_ylabel("Score", fontsize=8)
        ax.set_title(method, fontsize=8)
        for label, (start, end) in config.EVENTS.items():
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       alpha=0.2, color=colors.get(label, "#aaa"), label=label)
        ax.legend(fontsize=6, loc="upper left")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(f"E2 — Event time series: {region}", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# E4 — Temporal forecastability with bootstrap CI on AE-vs-PCA R² delta
# ---------------------------------------------------------------------------

def _make_lag_features(series: np.ndarray, lags: list[int]):
    max_lag = max(lags)
    X = np.column_stack([series[max_lag - l:-l if l > 0 else None] for l in lags])
    y = series[max_lag:]
    return X, y


def compute_forecastability(
    df: pd.DataFrame,
    region: str,
    lags: list[int] = config.LASSO_LAGS,
    train_frac: float = config.TRAIN_FRACTION,
    n_bootstrap: int = config.BOOTSTRAP_ITERS,
) -> pd.DataFrame:
    """
    Lasso(lags 1–4) one-step-ahead R²; bootstrap CI on R² delta vs best PCA.
    For AE3D, compare against best B3T (temporal PCA); for AE2D, vs best B3.
    """
    sub = df[df["region"] == region].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.sort_values("date")

    rng = np.random.default_rng(config.SEED)
    method_r2: dict[str, float] = {}
    method_residuals: dict[str, tuple] = {}

    for method in sorted(sub["method"].unique()):
        series = sub[sub["method"] == method].set_index("date")["score"].dropna()
        if len(series) < 20:
            continue
        X, y = _make_lag_features(series.values, lags)
        n_train = int(len(X) * train_frac)
        if n_train < len(lags) + 2:
            continue
        X_tr, X_te = X[:n_train], X[n_train:]
        y_tr, y_te = y[:n_train], y[n_train:]
        model = Lasso(alpha=0.1, max_iter=5000)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        method_r2[method] = float(r2_score(y_te, y_pred))
        method_residuals[method] = (y_te, y_pred)

    best_b3 = max((m for m in method_r2 if m.startswith("B3_pca")),
                  key=lambda m: method_r2[m], default=None)
    best_b3t = max((m for m in method_r2 if m.startswith("B3T_pca")),
                   key=lambda m: method_r2[m], default=None)

    results = []
    for method, r2 in method_r2.items():
        ci_low, ci_high, baseline = float("nan"), float("nan"), None
        if method.startswith("AE_3d") and best_b3t:
            baseline = best_b3t
        elif method.startswith("AE_2d") and best_b3:
            baseline = best_b3
        elif method.startswith("AE") and best_b3:  # legacy AE_l32 fallback
            baseline = best_b3

        if baseline and baseline != method:
            y_ae, y_ae_pred = method_residuals[method]
            y_pca, y_pca_pred = method_residuals[baseline]
            n = min(len(y_ae), len(y_pca))
            deltas = []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                d = r2_score(y_ae[:n][idx], y_ae_pred[:n][idx]) - \
                    r2_score(y_pca[:n][idx], y_pca_pred[:n][idx])
                deltas.append(d)
            ci_low = float(np.percentile(deltas, 2.5))
            ci_high = float(np.percentile(deltas, 97.5))

        results.append({
            "region": region, "method": method, "r2": r2,
            "ci_low_vs_baseline": ci_low, "ci_high_vs_baseline": ci_high,
            "baseline_method": baseline,
        })
    return pd.DataFrame(results)


def plot_forecastability(fore_df: pd.DataFrame) -> plt.Figure:
    regions = sorted(fore_df["region"].unique())
    fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 4))
    if len(regions) == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        r = fore_df[fore_df["region"] == region].sort_values("r2", ascending=False)
        colors = []
        for m in r["method"]:
            if m.startswith("AE_3d"):   colors.append("#1d3557")  # dark blue
            elif m.startswith("AE_2d"): colors.append("#2a9d8f")  # teal
            elif m.startswith("B3T"):   colors.append("#e76f51")  # orange
            elif m.startswith("B3"):    colors.append("#f4a261")  # light orange
            else:                       colors.append("#aaa")
        ax.barh(r["method"], r["r2"], color=colors)
        for _, row in r.iterrows():
            if not (np.isnan(row["ci_low_vs_baseline"]) or np.isnan(row["ci_high_vs_baseline"])):
                y_pos = list(r["method"]).index(row["method"])
                ax.plot([row["r2"] + row["ci_low_vs_baseline"],
                         row["r2"] + row["ci_high_vs_baseline"]],
                        [y_pos, y_pos], color="black", linewidth=2)
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("One-step-ahead R²")
        ax.set_title(region, fontsize=9)

    fig.suptitle("E4 — Temporal forecastability (CI: AE vs matched PCA)", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# E5 — Channel ablation
# ---------------------------------------------------------------------------

def plot_channel_ablation(fore_df: pd.DataFrame, region: str) -> plt.Figure:
    r = fore_df[fore_df["region"] == region].copy()
    ae_rows = r[r["method"].str.startswith("AE")].sort_values("r2", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(ae_rows["method"], ae_rows["r2"], color="#264653")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("One-step-ahead R²")
    ax.set_title(f"E5 — Channel ablation R²: {region}", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sanity: yearly drift
# ---------------------------------------------------------------------------

def plot_yearly_drift(df: pd.DataFrame, region: str) -> plt.Figure:
    sub = df[df["region"] == region].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub["year"] = sub["date"].dt.year
    yearly = sub.groupby(["year", "method"])["score"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    for m in sorted(yearly["method"].unique()):
        d = yearly[yearly["method"] == m]
        ax.plot(d["year"], d["score"], label=m, marker="o", markersize=3)
    ax.set_xlabel("Year"); ax.set_ylabel("Mean score")
    ax.set_title(f"Sanity: yearly mean score drift — {region}")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# E3 — Spatial coherence (Moran's I, numpy only)
# ---------------------------------------------------------------------------

def morans_i(values: np.ndarray, valid: np.ndarray) -> float:
    """Moran's I on a (H, W) array using rook (4-neighbour) contiguity."""
    flat_vals = values[valid].astype(np.float64)
    if flat_vals.size < 4:
        return float("nan")
    rows, cols = np.where(valid)
    n = len(rows)
    z = flat_vals - flat_vals.mean()

    coord_set = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}
    W = np.zeros((n, n), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            j = coord_set.get((r + dr, c + dc))
            if j is not None:
                W[i, j] = 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    W_row = W / row_sums
    numerator = float(z @ W_row @ z)
    denominator = float(z @ z)
    if denominator < 1e-12:
        return float("nan")
    return n * numerator / (W.sum() * denominator)


# ---------------------------------------------------------------------------
# Sanity: cloud-coverage confound + aggregation comparison
# ---------------------------------------------------------------------------

def plot_cloud_confound(df: pd.DataFrame, valid_fracs: pd.Series) -> plt.Figure:
    """Scatter score vs valid-pixel fraction per frame."""
    sub = df.copy()
    sub["date"] = pd.to_datetime(sub["date"])
    valid_fracs.index = pd.to_datetime(valid_fracs.index)
    merged = sub.merge(valid_fracs.rename("valid_frac"), left_on="date", right_index=True)
    methods = sorted(merged["method"].unique())

    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        m = merged[merged["method"] == method].dropna(subset=["score", "valid_frac"])
        ax.scatter(m["valid_frac"], m["score"], alpha=0.3, s=5)
        corr = float(m[["valid_frac", "score"]].corr().iloc[0, 1]) if len(m) > 2 else float("nan")
        ax.set_title(f"{method}\nr={corr:.2f}", fontsize=8)
        ax.set_xlabel("Valid-pixel fraction")
        ax.set_ylabel("Score")
    fig.suptitle("Sanity: cloud-coverage confound", fontsize=10)
    fig.tight_layout()
    return fig


def plot_aggregation_comparison(df: pd.DataFrame, region: str, method: str) -> plt.Figure:
    """Overlay mean / top_decile / max aggregations for one method in one region."""
    sub = df[(df["region"] == region) & (df["method"] == method)].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    fig, ax = plt.subplots(figsize=(12, 4))
    for agg in ["mean", "top_decile", "max"]:
        d = sub[sub["aggregation"] == agg].sort_values("date")
        if d.empty:
            continue
        ax.plot(d["date"], d["score"], label=agg, linewidth=0.8)
    ax.set_title(f"Aggregation comparison — {method} — {region}")
    ax.legend()
    fig.tight_layout()
    return fig

"""
Multi-step-ahead forecastability sweep.

Same Lasso(lags 1-4) protocol as E4, but predict y[t + h] instead of y[t].
Sweep h ∈ {1, 7, 14, 28, 56} days. h=1 reproduces existing E4 (1-day-ahead,
which is largely trivial given 8-day rolling composites). h ≥ 8 is the
informative regime since composite windows no longer overlap.

For each AE method, bootstrap 1000× the R² delta vs matched-k PCA at the
same lead time. AE_3d_* compared vs best B3T_pca_*_t4; AE_2d_* vs best B3_pca_*.

Outputs:
    outputs/figures/E4_multistep.parquet
    outputs/figures/E4_multistep.png  — R² vs lead time per method, per region

Usage:
    python scripts/08_multistep_forecast.py
    python scripts/08_multistep_forecast.py --leads 1 7 14 28 56
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.regions import REGIONS


LAGS = config.LASSO_LAGS  # [1, 2, 3, 4]
TRAIN_FRAC = config.TRAIN_FRACTION
N_BOOT = config.BOOTSTRAP_ITERS


def _make_h_step_features(series: np.ndarray, lags: list[int], h: int):
    """
    X[i] = [series[i + max_lag - l] for l in lags]
    y[i] = series[i + max_lag - 1 + h]

    With lags=[1,2,3,4] and h=1, this matches the existing 1-step-ahead protocol
    (predict from the 4 most recent past values).
    With h=7, predict 7 indices ahead of the most-recent input.
    """
    max_lag = max(lags)
    n = len(series)
    last = n - h  # last valid prediction target index
    if last <= max_lag:
        return None, None
    X_cols = [series[max_lag - l : last - h + (max_lag - l) + 1 - (max_lag - l)]
              for l in lags]
    # Simpler windowed construction:
    starts = []
    targets = []
    for i in range(max_lag, last):
        starts.append([series[i - l] for l in lags])
        targets.append(series[i + h - 1])
    X = np.asarray(starts, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    return X, y


def fit_eval(series: np.ndarray, lags: list[int], h: int, train_frac: float):
    X, y = _make_h_step_features(series, lags, h)
    if X is None or len(X) < (max(lags) + 4):
        return None
    n_train = int(len(X) * train_frac)
    if n_train < max(lags) + 2 or n_train >= len(X) - 2:
        return None
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    model = Lasso(alpha=0.1, max_iter=5000)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    r2 = float(r2_score(y_te, y_pred))
    return r2, (y_te, y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", default=str(config.SCORES_DIR))
    parser.add_argument("--out-dir", default=str(config.FIGURES_DIR))
    parser.add_argument("--aggregation", default="mean")
    parser.add_argument("--leads", type=int, nargs="+", default=[1, 7, 14, 28, 56],
                        help="Lead times in DAYS (cube is daily cadence).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_scores = Path(args.scores_dir) / "all_scores.parquet"
    print(f"Loading {all_scores} ...")
    df = pd.read_parquet(all_scores)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["aggregation"] == args.aggregation]
    df = df.sort_values(["region", "method", "date"])
    print(f"  {len(df):,} rows, {df['method'].nunique()} methods, {df['region'].nunique()} regions")

    rng = np.random.default_rng(config.SEED)
    region_names = [r.name for r in REGIONS]

    all_rows = []
    for region in region_names:
        sub = df[df["region"] == region]
        if sub.empty:
            continue
        method_series: dict[str, np.ndarray] = {}
        for method in sorted(sub["method"].unique()):
            s = sub[sub["method"] == method].set_index("date")["score"].dropna().sort_index().values
            if len(s) >= 64:
                method_series[method] = s

        for h in args.leads:
            r2: dict[str, float] = {}
            residuals: dict[str, tuple] = {}
            for method, series in method_series.items():
                out = fit_eval(series, LAGS, h, TRAIN_FRAC)
                if out is None:
                    continue
                r2[method], residuals[method] = out

            best_b3 = max((m for m in r2 if m.startswith("B3_pca")),
                          key=lambda m: r2[m], default=None)
            best_b3t = max((m for m in r2 if m.startswith("B3T_pca")),
                           key=lambda m: r2[m], default=None)

            for method, val in r2.items():
                ci_low = ci_high = float("nan")
                baseline = None
                if method.startswith("AE_3d") and best_b3t:
                    baseline = best_b3t
                elif method.startswith("AE_2d") and best_b3:
                    baseline = best_b3

                if baseline and baseline != method:
                    y_ae, y_ae_pred = residuals[method]
                    y_pca, y_pca_pred = residuals[baseline]
                    n = min(len(y_ae), len(y_pca))
                    deltas = np.empty(N_BOOT)
                    for b in range(N_BOOT):
                        idx = rng.integers(0, n, size=n)
                        d_ae = r2_score(y_ae[:n][idx], y_ae_pred[:n][idx])
                        d_pca = r2_score(y_pca[:n][idx], y_pca_pred[:n][idx])
                        deltas[b] = d_ae - d_pca
                    ci_low = float(np.percentile(deltas, 2.5))
                    ci_high = float(np.percentile(deltas, 97.5))

                all_rows.append({
                    "region": region, "method": method, "lead_days": h, "r2": val,
                    "ci_low_vs_baseline": ci_low, "ci_high_vs_baseline": ci_high,
                    "baseline_method": baseline,
                })

        # progress
        print(f"  done {region} ({len(method_series)} methods × {len(args.leads)} leads)")

    out_df = pd.DataFrame(all_rows)
    out_parquet = out_dir / "E4_multistep.parquet"
    out_df.to_parquet(out_parquet, index=False)
    print(f"\nWrote {out_parquet} ({len(out_df)} rows)")

    # Print decay summary for AE_3d_l32_t4_mae070 (headline method) per region
    print("\n" + "=" * 80)
    print("Lead-time decay for AE_3d_l32_t4_mae070 (headline method):")
    headline = out_df[out_df["method"] == "AE_3d_l32_t4_mae070"]
    if not headline.empty:
        pivot = headline.pivot(index="region", columns="lead_days", values="r2")
        print(pivot.to_string(float_format=lambda v: f"{v:+.3f}"))

    # Same table for matched PCA baseline
    print("\nMatched PCA (B3T_pca_*_t4 best) for reference:")
    pca_rows = out_df[out_df["method"].str.startswith("B3T_pca")]
    if not pca_rows.empty:
        # best per region per lead
        best_pca = pca_rows.loc[pca_rows.groupby(["region", "lead_days"])["r2"].idxmax()]
        pivot_pca = best_pca.pivot(index="region", columns="lead_days", values="r2")
        print(pivot_pca.to_string(float_format=lambda v: f"{v:+.3f}"))

    # Plot: per region, R² vs lead_days for top methods
    methods_to_plot = [
        "AE_3d_l32_t4_mae070", "AE_3d_l32_t4_mae040", "AE_3d_l32_t4_mae030",
        "AE_3d_l32_t4", "AE_2d_l32_mae070", "AE_2d_l32",
        "B3T_pca_k32_t4", "B3_pca_k32", "B1_chla_zscore", "B2_multivar_zscore",
    ]
    colors = {
        "AE_3d_l32_t4_mae070": "#1d3557", "AE_3d_l32_t4_mae040": "#457b9d",
        "AE_3d_l32_t4_mae030": "#a8dadc", "AE_3d_l32_t4": "#2a9d8f",
        "AE_2d_l32_mae070": "#264653", "AE_2d_l32": "#8ab17d",
        "B3T_pca_k32_t4": "#e76f51", "B3_pca_k32": "#f4a261",
        "B1_chla_zscore": "#bbb", "B2_multivar_zscore": "#999",
    }

    fig, axes = plt.subplots(1, len(region_names), figsize=(4 * len(region_names), 4), sharey=True)
    if len(region_names) == 1:
        axes = [axes]
    for ax, region in zip(axes, region_names):
        r_df = out_df[out_df["region"] == region]
        for method in methods_to_plot:
            m = r_df[r_df["method"] == method].sort_values("lead_days")
            if m.empty:
                continue
            ax.plot(m["lead_days"], m["r2"], label=method, marker="o",
                    color=colors.get(method, "#aaa"), linewidth=1.5, markersize=4)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Lead time (days)")
        ax.set_title(region, fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(args.leads); ax.set_xticklabels(args.leads)
    axes[0].set_ylabel("R²")
    axes[-1].legend(fontsize=5, loc="lower left", bbox_to_anchor=(1.0, 0.0))
    fig.suptitle("Multi-step-ahead forecastability  (cube is daily cadence; lead in DAYS)", fontsize=10)
    fig.tight_layout()
    png = out_dir / "E4_multistep.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {png}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals for DATect Paper Metrics

Computes 95% bootstrap CIs on R², MAE, RMSE for each site and overall
from the cached retrospective results. No model retraining needed.

Usage (runs locally):
    python3 paper_bootstrap_ci.py

Output: paper_bootstrap_results.json (paste into LaTeX tables)
"""

import numpy as np
import pandas as pd
import json
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

N_BOOTSTRAP = 2000
SEED = 42
CACHE_PATH = "cache/retrospective/regression_ensemble.parquet"


def bootstrap_metrics(y_true, y_pred, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Compute bootstrap 95% CIs for R², MAE, RMSE."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    r2s, maes, rmses = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        # R² can be undefined if bootstrap sample has zero variance
        if np.std(yt) == 0:
            continue
        r2s.append(r2_score(yt, yp))
        maes.append(mean_absolute_error(yt, yp))
        rmses.append(np.sqrt(mean_squared_error(yt, yp)))

    return {
        'r2': float(r2_score(y_true, y_pred)),
        'r2_ci_lo': float(np.percentile(r2s, 2.5)),
        'r2_ci_hi': float(np.percentile(r2s, 97.5)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mae_ci_lo': float(np.percentile(maes, 2.5)),
        'mae_ci_hi': float(np.percentile(maes, 97.5)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'rmse_ci_lo': float(np.percentile(rmses, 2.5)),
        'rmse_ci_hi': float(np.percentile(rmses, 97.5)),
        'n': n,
    }


def format_metric(val, ci_lo, ci_hi, decimals=3):
    """Format as: 0.457 [0.392, 0.518]"""
    return f"{val:.{decimals}f} [{ci_lo:.{decimals}f}, {ci_hi:.{decimals}f}]"


def main():
    df = pd.read_parquet(CACHE_PATH)
    y_true = df['actual_da'].values
    y_pred = df['predicted_da'].values

    results = {}

    # Overall
    overall = bootstrap_metrics(y_true, y_pred)
    results['Overall'] = overall

    # Per-site
    for site in sorted(df['site'].unique()):
        mask = df['site'] == site
        site_true = df.loc[mask, 'actual_da'].values
        site_pred = df.loc[mask, 'predicted_da'].values
        results[site] = bootstrap_metrics(site_true, site_pred)

    # Print LaTeX-ready table
    print("=" * 90)
    print("Bootstrap 95% CIs for Paper (n_bootstrap={})".format(N_BOOTSTRAP))
    print("=" * 90)
    print()
    print(f"{'Site':<18} {'N':>5}  {'R²':>28}  {'MAE (µg/g)':>28}  {'RMSE (µg/g)':>28}")
    print("-" * 115)

    # Print per-site first, then overall
    site_order = [
        'Twin Harbors', 'Copalis', 'Kalaloch', 'Quinault', 'Long Beach',
        'Clatsop Beach', 'Coos Bay', 'Newport', 'Gold Beach', 'Cannon Beach',
    ]

    for site in site_order:
        if site not in results:
            continue
        r = results[site]
        r2_str = format_metric(r['r2'], r['r2_ci_lo'], r['r2_ci_hi'])
        mae_str = format_metric(r['mae'], r['mae_ci_lo'], r['mae_ci_hi'], 2)
        rmse_str = format_metric(r['rmse'], r['rmse_ci_lo'], r['rmse_ci_hi'], 2)
        print(f"{site:<18} {r['n']:>5}  {r2_str:>28}  {mae_str:>28}  {rmse_str:>28}")

    print("-" * 115)
    r = results['Overall']
    r2_str = format_metric(r['r2'], r['r2_ci_lo'], r['r2_ci_hi'])
    mae_str = format_metric(r['mae'], r['mae_ci_lo'], r['mae_ci_hi'], 2)
    rmse_str = format_metric(r['rmse'], r['rmse_ci_lo'], r['rmse_ci_hi'], 2)
    print(f"{'Overall':<18} {r['n']:>5}  {r2_str:>28}  {mae_str:>28}  {rmse_str:>28}")

    # Save JSON
    with open('paper_bootstrap_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("Saved to paper_bootstrap_results.json")


if __name__ == "__main__":
    main()

"""
autocorrelation_diagnostic.py — Per-site DA predictability ceiling

Computes the lag-1 autocorrelation of raw DA measurements for each site.
The squared autocorrelation (rho²) is the theoretical upper bound on R² for
a **persistence-only** (last-value) forecast.  Models that use environmental
features (SST, BEUTI, PDO, discharge, etc.) can and do exceed this bound —
rho² is a floor on model potential (what persistence alone captures), not an
upper bound on what is achievable with additional predictors.

This is a read-only diagnostic — it does not affect predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg


def compute_autocorrelation_ceiling(raw_da: pd.DataFrame) -> Dict[str, dict]:
    """
    Compute lag-1 autocorrelation and R² ceiling for each site.

    Parameters
    ----------
    raw_da : DataFrame with columns ['site', 'date', 'da_raw']
        Raw DA measurements (not interpolated).

    Returns
    -------
    dict mapping site -> {
        'n': int,                   # number of real measurements
        'autocorr_lag1': float,     # lag-1 autocorrelation of raw DA series
        'r2_ceiling': float,        # rho² — theoretical max R² for one-step forecast
        'mean_da': float,
        'std_da': float,
        'pct_nonzero': float,       # fraction of measurements > 0
    }
    """
    results = {}
    raw_da = raw_da.copy()
    raw_da["date"] = pd.to_datetime(raw_da["date"])

    for site in sorted(raw_da["site"].unique()):
        site_df = (
            raw_da[raw_da["site"] == site]
            .sort_values("date")
            .dropna(subset=["da_raw"])
        )
        vals = site_df["da_raw"].values.astype(float)
        n = len(vals)

        if n < 4:
            results[site] = {
                "n": n, "autocorr_lag1": None, "r2_ceiling": None,
                "mean_da": None, "std_da": None, "pct_nonzero": None,
            }
            continue

        # Lag-1 autocorrelation on the raw series (not de-meaned — matches
        # what a persistence baseline effectively measures)
        rho = float(np.corrcoef(vals[:-1], vals[1:])[0, 1])
        r2_ceil = round(rho ** 2, 4)

        results[site] = {
            "n": n,
            "autocorr_lag1": round(rho, 4),
            "r2_ceiling": r2_ceil,
            "mean_da": round(float(np.mean(vals)), 2),
            "std_da": round(float(np.std(vals)), 2),
            "pct_nonzero": round(float(np.mean(vals > 0)), 4),
        }

    return results


def print_ceiling_table(results: Dict[str, dict], current_r2: Dict[str, float] | None = None):
    """Print a formatted ceiling table, optionally comparing to current model R²."""
    WA = {"Kalaloch", "Quinault", "Copalis", "Twin Harbors", "Long Beach"}

    print("\n" + "=" * 90)
    print("PER-SITE PREDICTABILITY CEILING (lag-1 autocorrelation²)")
    print("=" * 90)
    header = f"  {'Site':<18s}  {'N':>4s}  {'ρ lag-1':>8s}  {'R² ceil':>8s}  {'Mean DA':>8s}  {'Std DA':>8s}  {'%>0':>6s}"
    if current_r2:
        header += f"  {'Cur R²':>8s}  {'Gap':>7s}"
    print(header)
    print("  " + "-" * 86)

    for site in cfg.SITES:
        m = results.get(site, {})
        if m.get("autocorr_lag1") is None:
            print(f"  {site:<18s}  {'n/a':>4s}  — insufficient data —")
            continue
        region = "WA" if site in WA else "OR"
        row = (
            f"  {site:<18s}  {m['n']:>4d}  {m['autocorr_lag1']:>8.4f}"
            f"  {m['r2_ceiling']:>8.4f}  {m['mean_da']:>8.2f}  {m['std_da']:>8.2f}"
            f"  {m['pct_nonzero']:>5.1%}"
        )
        if current_r2 and site in current_r2 and current_r2[site] is not None:
            cur = current_r2[site]
            gap = cur - m["r2_ceiling"]
            row += f"  {cur:>8.4f}  {gap:>+7.4f}"
        print(row)

    # Summary
    wa_ceil = [m["r2_ceiling"] for s, m in results.items() if s in WA and m.get("r2_ceiling") is not None]
    or_ceil = [m["r2_ceiling"] for s, m in results.items() if s not in WA and m.get("r2_ceiling") is not None]
    if wa_ceil:
        print(f"\n  WA mean R² ceiling: {np.mean(wa_ceil):.4f}")
    if or_ceil:
        print(f"  OR mean R² ceiling: {np.mean(or_ceil):.4f}")
    print("=" * 90)
    print("  ρ lag-1: Pearson correlation of consecutive raw DA observations")
    print("  R² ceil: ρ² — theoretical max R² for a persistence-only forecast")
    print("  Gap: current model R² minus ceiling (positive = env features add signal beyond persistence)")


if __name__ == "__main__":
    from forecasting.raw_data_forecaster import load_raw_da_measurements
    raw = load_raw_da_measurements()
    results = compute_autocorrelation_ceiling(raw)
    print_ceiling_table(results)

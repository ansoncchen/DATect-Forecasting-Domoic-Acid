"""
Chain 1: Lagged Pseudo-nitzschia features for high-N sites.

DATect already has `pn` (cell count) in final_output.parquet. Only Kalaloch
uses it in feature_subset; 9 other sites leave free signal on the table.
Correlation analysis (§17 of OAD_INTEGRATION_RESULTS.md) shows lag 2-4w PN
correlates with shellfish DA at r=+0.18 to +0.31 at the 4 high-N sites.

This chain adds:
  - pn_log              (log1p of concurrent PN; already exists as pn_log if Kalaloch enabled)
  - pn_log_lag1w        (1-week lag, leak-safe)
  - pn_log_lag2w        (2-week lag — strongest correlation per §17)
  - pn_log_lag4w        (4-week lag)
  - pn_log_30day_mean   (mean over [R-35d, R-LEAK_SHIFT_DAYS])
  - pn_log_30day_max    (max over same window — captures spikes)
  - weeks_since_pn_spike (weeks since pn > 100,000 cells/L)

Target sites: Twin Harbors, Long Beach, Quinault, Copalis, Kalaloch
(the 5 sites with r > 0.10 at any lag in §17 table)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CHAIN_NAME = "lagged_pn"
NEW_FEATURES = [
    "pn_log",
    "pn_log_lag1w",
    "pn_log_lag2w",
    "pn_log_lag4w",
    "pn_log_30day_mean",
    "pn_log_30day_max",
    "weeks_since_pn_spike",
]
TARGET_SITES = ["Twin Harbors", "Long Beach", "Quinault", "Copalis", "Kalaloch"]
LEAK_SHIFT_DAYS = 0  # PN is measured at the beach so no satellite-composite shift needed
PN_SPIKE_THRESHOLD = 100_000  # cells/L; matches CLAUDE.md spike threshold spirit


def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["site", "date"]).reset_index(drop=True)

    # Compute per-site
    pieces = []
    for site, sub in df.groupby("site", group_keys=False):
        sub = sub.sort_values("date").reset_index(drop=True).copy()
        # log1p handles zeros gracefully (log1p(0) = 0); raw PN ranges 0 to 4.7M
        sub["pn_log"] = np.log1p(sub["pn"].astype(float))
        # Weekly rows are 7-day spaced — use observation-order lags (DATect pattern)
        for lag_obs, name in [(1, "pn_log_lag1w"), (2, "pn_log_lag2w"), (4, "pn_log_lag4w")]:
            sub[name] = sub["pn_log"].shift(lag_obs)
        # 30-day rolling (window = 4-5 obs, since weekly cadence)
        # Use shift(1) to avoid concurrent leakage; mean and max over past 4 obs
        roll = sub["pn_log"].shift(1).rolling(window=4, min_periods=2)
        sub["pn_log_30day_mean"] = roll.mean()
        sub["pn_log_30day_max"] = roll.max()
        # weeks_since_pn_spike: count weeks back to last pn > threshold
        is_spike = (sub["pn"] > PN_SPIKE_THRESHOLD).astype(int)
        # shift(1) so the count doesn't see the current week
        is_spike_shifted = is_spike.shift(1).fillna(0)
        # For each row, find the index of last 1 in is_spike_shifted[:row]
        wsps = np.full(len(sub), np.nan)
        last_spike_idx = -1
        for i in range(len(sub)):
            if i > 0 and is_spike_shifted.iloc[i - 1] == 1:
                last_spike_idx = i - 1
            if last_spike_idx >= 0:
                wsps[i] = i - last_spike_idx  # weeks (= rows) since spike
        sub["weeks_since_pn_spike"] = wsps
        pieces.append(sub)

    out = pd.concat(pieces).sort_values(["site", "date"]).reset_index(drop=True)
    # Convert date back to MM/DD/YYYY string to match existing schema
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%m/%d/%Y")
    return out

"""
Chain 2: BEUTI derivatives — temporal context for the existing `beuti` feature.

DATect has `beuti` (Bakun Upwelling Index) as a single instantaneous value.
Moore et al. 2021 (§14 of writeup) shows that BEUTI's TEMPORAL pattern
(5+ days of northerly winds → nutrient shoaling → bloom precursor) is what
drives Pn blooms, not the raw value. This chain adds derivatives.

New features (computed per site, weekly-shifted to be leak-safe):
  - beuti_lag1w, beuti_lag2w, beuti_lag4w  (raw lagged values)
  - beuti_14day_delta        (current - 2-weeks-ago, captures upwelling pulse)
  - beuti_30day_mean         (4-week rolling mean)
  - beuti_30day_anom         (mean - DOY climatology)
  - beuti_30day_max          (rolling max)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CHAIN_NAME = "beuti_derivatives"
NEW_FEATURES = [
    "beuti_lag1w",
    "beuti_lag2w",
    "beuti_lag4w",
    "beuti_14day_delta",
    "beuti_30day_mean",
    "beuti_30day_anom",
    "beuti_30day_max",
]
TARGET_SITES = None  # all 10 sites
LEAK_SHIFT_DAYS = 0  # beuti is a single value per date, no composite-window issue


def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["site", "date"]).reset_index(drop=True)

    pieces = []
    for site, sub in df.groupby("site", group_keys=False):
        sub = sub.sort_values("date").reset_index(drop=True).copy()
        sub["beuti_lag1w"] = sub["beuti"].shift(1)
        sub["beuti_lag2w"] = sub["beuti"].shift(2)
        sub["beuti_lag4w"] = sub["beuti"].shift(4)
        sub["beuti_14day_delta"] = sub["beuti"] - sub["beuti"].shift(2)
        # 30-day rolling, shifted 1 to avoid concurrent leak
        roll = sub["beuti"].shift(1).rolling(window=4, min_periods=2)
        sub["beuti_30day_mean"] = roll.mean()
        sub["beuti_30day_max"] = roll.max()
        # DOY climatology: per-site mean BEUTI at each day-of-year across all years,
        # then subtract from current 30-day mean for the anomaly
        sub["_doy"] = sub["date"].dt.dayofyear
        doy_clim = sub.groupby("_doy")["beuti"].transform("mean")
        sub["beuti_30day_anom"] = sub["beuti_30day_mean"] - doy_clim
        sub = sub.drop(columns=["_doy"])
        pieces.append(sub)

    out = pd.concat(pieces).sort_values(["site", "date"]).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%m/%d/%Y")
    return out

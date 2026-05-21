"""
Chain 3: NEMO mooring in-situ features for WA sites.

Loads ChaBa_pCO2ALLdata.mat (23,483 rows, 2010-2023+) with NEMO subsurface
state: SST, salinity (SSS), dissolved O2 (DOmgl), chlorophyll (chl, in-situ
fluorometric), turbidity (turb), pCO2, pH (mostly missing), density (sgth),
pressure (pr).

NEMO sits at ~18m depth, offshore WA shelf (24 km W/NW of La Push). It's the
upstream source for ALL WA beaches per Moore et al. 2021. So these features
apply to: Kalaloch, Quinault, Copalis, Twin Harbors, Long Beach, Clatsop
Beach, Cannon Beach. (Not OR sites — too far south.)

New features per (date, site in WA region):
  - nemo_sst              (daily mean, lag-safe)
  - nemo_chl              (in-situ chlorophyll)
  - nemo_DO               (dissolved oxygen mg/L)
  - nemo_SSS              (salinity)
  - nemo_pCO2             (xCO2water)
  - nemo_turb             (turbidity)
  - nemo_sst_14day_delta  (recent change — upwelling/wave proxy)
  - nemo_chl_30day_max    (recent bloom intensity at source)

LEAK_SHIFT_DAYS = 0 because NEMO measurements are in-situ instantaneous, not
satellite composites. No 8-day-composite lag adjustment needed.

Path to .mat: /Users/ansonchen/Downloads/ChaBa_pCO2ALLdata.mat
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

CHAIN_NAME = "nemo_mooring"
NEW_FEATURES = [
    "nemo_sst",
    "nemo_chl",
    "nemo_DO",
    "nemo_SSS",
    "nemo_pCO2",
    "nemo_turb",
    "nemo_sst_14day_delta",
    "nemo_chl_30day_max",
]
TARGET_SITES = ["Kalaloch", "Quinault", "Copalis", "Twin Harbors",
                "Long Beach", "Clatsop Beach", "Cannon Beach"]
LEAK_SHIFT_DAYS = 0

MAT_PATH_CANDIDATES = [
    "/Users/ansonchen/Downloads/ChaBa_pCO2ALLdata.mat",
    "data/raw/ChaBa_pCO2ALLdata.mat",  # if copied into repo
]


def _load_nemo_daily() -> pd.DataFrame:
    from scipy.io import loadmat
    src = next((p for p in MAT_PATH_CANDIDATES if Path(p).exists()), None)
    if src is None:
        raise FileNotFoundError(f"NEMO .mat not found at any of {MAT_PATH_CANDIDATES}")
    m = loadmat(src, squeeze_me=True)
    pco2 = m["pco2"].item()
    fields = ["Date", "pr", "xCO2water", "xCO2air", "pH_tot", "SSS", "SST",
              "chl", "turb", "DOuM", "DOmgl", "DOumolperkg", "sgth", "dtnum"]
    arrs = dict(zip(fields, pco2))
    df = pd.DataFrame({
        "datetime": pd.to_datetime(arrs["Date"]),
        "nemo_sst": arrs["SST"].astype(float),
        "nemo_chl": arrs["chl"].astype(float),
        "nemo_DO":  arrs["DOmgl"].astype(float),
        "nemo_SSS": arrs["SSS"].astype(float),
        "nemo_pCO2": arrs["xCO2water"].astype(float),
        "nemo_turb": arrs["turb"].astype(float),
    })
    # Aggregate to daily means (mooring is 3-hourly)
    df["date"] = df["datetime"].dt.normalize()
    daily = df.groupby("date").agg({
        "nemo_sst": "mean", "nemo_chl": "mean", "nemo_DO": "mean",
        "nemo_SSS": "mean", "nemo_pCO2": "mean", "nemo_turb": "mean",
    }).reset_index()
    return daily


def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    nemo = _load_nemo_daily()
    df = df_in.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["site", "date"]).reset_index(drop=True)

    # Merge on date (left join). NEMO is one time series shared across all sites,
    # but only TARGET_SITES will have it actively used per per_site_models.py.
    out = df.merge(nemo, on="date", how="left")

    # Derived features
    pieces = []
    for site, sub in out.groupby("site", group_keys=False):
        sub = sub.sort_values("date").reset_index(drop=True).copy()
        sub["nemo_sst_14day_delta"] = sub["nemo_sst"] - sub["nemo_sst"].shift(2)
        sub["nemo_chl_30day_max"] = sub["nemo_chl"].shift(1).rolling(window=4, min_periods=2).max()
        pieces.append(sub)
    out = pd.concat(pieces).sort_values(["site", "date"]).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%m/%d/%Y")
    return out

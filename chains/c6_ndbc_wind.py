"""
Chain 6: NDBC wind features for WA sites (Moore et al. 2021 mechanism).

Moore et al. 2021 (§14 of writeup) identifies the bloom precursor as 5+ days
of upwelling-favorable northerly winds → nutrient shoaling → bloom. The wind
data lives at NDBC buoy 46041 (Cape Elizabeth, WA), free, hourly back to ~1987.

Wind convention (NDBC standard meteorological):
  WDIR = direction wind COMES FROM, degrees clockwise from true north
  WSPD = wind speed, m/s
  Upwelling-favorable on PNW coast = wind FROM the north blowing south:
    → WDIR near 0° / 360°
    → v_component = -WSPD * cos(WDIR rad) is NEGATIVE (southward)

Features computed per weekly DATect row R (with leak-safe lag):
  - wind_v_30day_cum_neg     : ∫ max(0, -v_wind) dt over past 30 days
                               ("cumulative upwelling stress")
  - wind_v_14day_cum_neg     : same, 14 days
  - wind_v_7day_cum_neg      : same, 7 days
  - wind_v_14day_mean        : raw mean v_wind over past 14d (sign matters)
  - wind_reversal_count_30d  : count of N↔S wind transitions in past 30d
  - wind_speed_30day_mean    : storm/internal-wave proxy
  - wind_speed_30day_max     : storm intensity proxy

Coverage: 1987-present at 46041; mapped to DATect's 2002-2024 grid.

Note: ndbc data fetched via the simple historical URL pattern. Cached locally.
"""
from __future__ import annotations

from pathlib import Path
import io
import urllib.request
import gzip

import numpy as np
import pandas as pd

CHAIN_NAME = "ndbc_wind"
NEW_FEATURES = [
    "wind_v_7day_cum_neg",
    "wind_v_14day_cum_neg",
    "wind_v_30day_cum_neg",
    "wind_v_14day_mean",
    "wind_reversal_count_30d",
    "wind_speed_30day_mean",
    "wind_speed_30day_max",
]
# Apply to all sites — wind is regional, all 10 PNW beaches feel similar forcing
TARGET_SITES = None
LEAK_SHIFT_DAYS = 7  # buoy data is hourly; weekly grid uses 7-day shift to align with
                     # DATect's anchor convention. No 8-day composite issue here.

CACHE = Path("data/raw/ndbc_46041_wind.parquet")
NDBC_URL = ("https://www.ndbc.noaa.gov/view_text_file.php?"
            "filename=46041h{year}.txt.gz&dir=data/historical/stdmet/")


def _fetch_ndbc_year(year: int) -> pd.DataFrame | None:
    """Fetch and parse one year of NDBC standard met data. Returns None on failure."""
    url = NDBC_URL.format(year=year)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read()
    except Exception as e:
        print(f"  WARN: NDBC fetch failed for {year}: {e}")
        return None
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
            text = gz.read().decode("utf-8", errors="replace")
    except Exception:
        text = raw.decode("utf-8", errors="replace")
    # First two rows are header + units; skip them
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", skiprows=[1], low_memory=False)
    # Older years use YY for 2-digit year; newer use #YY
    yy_col = "#YY" if "#YY" in df.columns else ("YY" if "YY" in df.columns else "YYYY")
    if yy_col not in df.columns:
        return None
    yy = pd.to_numeric(df[yy_col], errors="coerce").astype("Int64")
    if yy.dropna().max() < 100:
        yy = yy.where(yy >= 50, yy + 100) + 1900  # 2-digit year → 4-digit
    def _col(name, default_val=0):
        if name not in df.columns:
            return pd.Series([default_val] * len(df))
        return pd.to_numeric(df[name], errors="coerce").fillna(default_val)
    mm = _col("MM", 1)
    dd = _col("DD", 1)
    hh = _col("hh", 0)
    mi = _col("mm", 0)
    df["datetime"] = pd.to_datetime(
        {"year": yy, "month": mm, "day": dd, "hour": hh, "minute": mi},
        errors="coerce",
    )
    wdir = pd.to_numeric(df.get("WDIR", df.get("WD")), errors="coerce")
    wspd = pd.to_numeric(df.get("WSPD"), errors="coerce")
    # Sentinel: 999 = missing for WDIR; 99 = missing for WSPD
    wdir = wdir.where((wdir >= 0) & (wdir <= 360))
    wspd = wspd.where((wspd >= 0) & (wspd < 99))
    out = pd.DataFrame({
        "datetime": df["datetime"],
        "wdir": wdir,
        "wspd": wspd,
    }).dropna(subset=["datetime"])
    return out


def _load_or_fetch_wind() -> pd.DataFrame:
    """Return concatenated multi-year NDBC 46041 wind dataframe. Cached."""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    print(f"  Fetching NDBC 46041 wind 2002-2024 (no cache at {CACHE})...")
    frames = []
    for year in range(2002, 2025):
        df = _fetch_ndbc_year(year)
        if df is not None and len(df) > 0:
            frames.append(df)
            print(f"    {year}: {len(df)} rows")
    if not frames:
        raise RuntimeError("No NDBC wind data fetched")
    out = pd.concat(frames).sort_values("datetime").drop_duplicates("datetime")
    # Compute v-component once (southward = negative v = upwelling-favorable)
    out["v_wind"] = -out["wspd"] * np.cos(np.radians(out["wdir"]))
    out.to_parquet(CACHE, index=False)
    print(f"  Cached {len(out)} hourly rows to {CACHE}")
    return out


def _aggregate_daily(wind_hourly: pd.DataFrame) -> pd.DataFrame:
    """Daily means + max-speed + transition count from hourly data."""
    w = wind_hourly.copy()
    w["date"] = w["datetime"].dt.normalize()
    w["v_neg"] = (-w["v_wind"]).clip(lower=0)  # only southward (upwelling) component
    # Sign of v: positive when v_wind>=0 (northward, downwelling), negative otherwise
    w["v_sign"] = np.sign(w["v_wind"]).fillna(0).astype(int)
    w["v_sign_change"] = (w["v_sign"] != w["v_sign"].shift(1)).astype(int)
    daily = w.groupby("date").agg(
        v_wind_mean=("v_wind", "mean"),
        v_neg_sum=("v_neg", "sum"),       # cumulative upwelling stress (hourly·m/s)
        wspd_mean=("wspd", "mean"),
        wspd_max=("wspd", "max"),
        n_reversals=("v_sign_change", "sum"),
    ).reset_index()
    return daily


def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    wind_hourly = _load_or_fetch_wind()
    daily = _aggregate_daily(wind_hourly).sort_values("date").reset_index(drop=True)

    # Rolling windows on the daily series — anchor at (R - LEAK_SHIFT_DAYS)
    daily.set_index("date", inplace=True)
    daily["wind_v_7day_cum_neg"]   = daily["v_neg_sum"].rolling("7D").sum()
    daily["wind_v_14day_cum_neg"]  = daily["v_neg_sum"].rolling("14D").sum()
    daily["wind_v_30day_cum_neg"]  = daily["v_neg_sum"].rolling("30D").sum()
    daily["wind_v_14day_mean"]     = daily["v_wind_mean"].rolling("14D").mean()
    daily["wind_reversal_count_30d"] = daily["n_reversals"].rolling("30D").sum()
    daily["wind_speed_30day_mean"]   = daily["wspd_mean"].rolling("30D").mean()
    daily["wind_speed_30day_max"]    = daily["wspd_max"].rolling("30D").max()
    daily = daily.reset_index()

    # Join onto DATect rows with leak shift
    df = df_in.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["_wind_lookup_date"] = df["date"] - pd.Timedelta(days=LEAK_SHIFT_DAYS)
    daily_slim = daily[["date"] + NEW_FEATURES].rename(columns={"date": "_wind_lookup_date"})
    out = df.merge(daily_slim, on="_wind_lookup_date", how="left").drop(columns=["_wind_lookup_date"])
    out["date"] = out["date"].dt.strftime("%m/%d/%Y")
    return out

"""
Raw Data Forecasting Utilities
==============================

Builds training datasets that use ONLY real raw DA measurements (no interpolation)
combined with environmental features from the processed dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import os
import pandas as pd
import numpy as np

import config
from .raw_data_processor import RawDataProcessor as DataProcessor
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RawForecastConfig:
    lags: Iterable[int] = tuple(config.LAG_FEATURES)
    max_date_diff_days: int = 14


def _normalize_site_name(site_key: str) -> str:
    return site_key.replace("-da", "").replace("_da", "").replace("-", " ").replace("_", " ").title()


def load_raw_da_measurements() -> pd.DataFrame:
    """
    Load raw DA measurements from CSVs.

    Returns: DataFrame with columns [date, site, da_raw]
    """
    raw_measurements = []
    for site_key, file_path in config.ORIGINAL_DA_FILES.items():
        if not os.path.exists(file_path):
            logger.warning("Raw DA file missing: %s", file_path)
            continue

        site_name = _normalize_site_name(site_key)
        df = pd.read_csv(file_path)

        date_col = None
        da_col = None
        if "CollectDate" in df.columns:
            date_col = "CollectDate"
        elif all(col in df.columns for col in ["Harvest Month", "Harvest Date", "Harvest Year"]):
            df["CombinedDateStr"] = (
                df["Harvest Month"].astype(str) + " "
                + df["Harvest Date"].astype(str) + ", "
                + df["Harvest Year"].astype(str)
            )
            df["ParsedDate"] = pd.to_datetime(df["CombinedDateStr"], format="%B %d, %Y", errors="coerce")
            date_col = "ParsedDate"

        if "Domoic Result" in df.columns:
            da_col = "Domoic Result"
        elif "Domoic Acid" in df.columns:
            da_col = "Domoic Acid"

        if date_col is None or da_col is None:
            logger.warning("Unknown raw DA schema for %s", file_path)
            continue

        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["da_raw"] = pd.to_numeric(df[da_col], errors="coerce")
        df["site"] = site_name

        valid_df = df.dropna(subset=["date", "da_raw"])
        valid_df = valid_df[valid_df["da_raw"] >= 0]
        raw_measurements.append(valid_df[["date", "site", "da_raw"]])

    if not raw_measurements:
        raise ValueError("No raw DA measurements could be loaded.")

    all_raw = pd.concat(raw_measurements, ignore_index=True)
    all_raw = all_raw.sort_values(["site", "date"]).reset_index(drop=True)
    return all_raw


def aggregate_raw_to_weekly(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw measurements to weekly (Monday) buckets using MAX.
    This preserves raw measurements but aligns to the weekly cadence.
    """
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week_date"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    weekly = (
        df.groupby(["site", "week_date"], as_index=False)["da_raw"]
        .max()
        .rename(columns={"week_date": "date"})
    )
    return weekly


def build_raw_feature_frame(
    processed_data: pd.DataFrame,
    raw_weekly: pd.DataFrame,
    config_override: Optional[RawForecastConfig] = None,
) -> pd.DataFrame:
    """
    Merge raw DA (weekly) into processed environmental features and add raw lags.
    """
    cfg = config_override or RawForecastConfig()
    base = processed_data.copy()
    base["date"] = pd.to_datetime(base["date"])

    raw_weekly = raw_weekly.copy()
    raw_weekly["date"] = pd.to_datetime(raw_weekly["date"])

    merged = base.merge(raw_weekly, on=["site", "date"], how="left")

    # Add persistence features from raw measurements (no interpolation)
    merged = merged.sort_values(["site", "date"])
    merged["last_observed_da_raw"] = merged.groupby("site")["da_raw"].ffill()
    last_obs_date = merged["date"].where(merged["da_raw"].notna())
    last_obs_date = last_obs_date.groupby(merged["site"]).ffill()
    merged["weeks_since_last_raw"] = (merged["date"] - last_obs_date).dt.days / 7.0
    merged["weeks_since_last_raw"] = merged["weeks_since_last_raw"].fillna(999.0)

    last_spike_date = merged["date"].where(merged["da_raw"] > config.SPIKE_THRESHOLD)
    last_spike_date = last_spike_date.groupby(merged["site"]).ffill()
    merged["weeks_since_last_spike"] = (merged["date"] - last_spike_date).dt.days / 7.0
    merged["weeks_since_last_spike"] = merged["weeks_since_last_spike"].fillna(999.0)

    if config.USE_ROLLING_FEATURES:
        rolling_windows = (4, 8, 12)
        shifted = merged.groupby("site")["last_observed_da_raw"].shift(1)
        for window in rolling_windows:
            rolling_group = shifted.groupby(merged["site"]).rolling(window, min_periods=1)
            merged[f"raw_obs_roll_mean_{window}"] = (
                rolling_group.mean().reset_index(level=0, drop=True)
            )
            merged[f"raw_obs_roll_std_{window}"] = (
                rolling_group.std().reset_index(level=0, drop=True)
            )
            merged[f"raw_obs_roll_max_{window}"] = (
                rolling_group.max().reset_index(level=0, drop=True)
            )

    processor = DataProcessor()
    merged = processor.create_raw_lag_features(
        merged, group_col="site", value_col="da_raw", lags=list(cfg.lags)
    )
    # Observation-order diffs (da_raw_prev_obs_diff_1_2) are now computed
    # inside create_raw_lag_features.  Old grid-shift lag diffs removed.
    return merged


def get_site_training_frame(
    feature_frame: pd.DataFrame,
    site: str,
    anchor_date: pd.Timestamp,
    min_training_samples: int = 10,
) -> Optional[pd.DataFrame]:
    """
    Select training rows for a site using only real raw measurements.
    """
    anchor_date = pd.Timestamp(anchor_date)
    site_data = feature_frame[feature_frame["site"] == site].copy()
    site_data = site_data.sort_values("date")
    train_data = site_data[site_data["date"] <= anchor_date].copy()
    train_data = train_data.dropna(subset=["da_raw"])
    if len(train_data) < min_training_samples:
        return None
    return train_data


def recompute_test_row_persistence_features(
    test_row: pd.DataFrame,
    train_data: pd.DataFrame,
    spike_threshold: float,
) -> pd.DataFrame:
    """
    Overwrite persistence features in test_row using only data from train_data
    (date <= anchor_date) to prevent target leakage.

    Only these features can leak the target:
    - last_observed_da_raw: ffill includes target if measurement exists at test_date
    - weeks_since_last_raw: would be 0 if target exists
    - weeks_since_last_spike: would be 0 if target is a spike

    NOTE: Rolling features (raw_obs_roll_*) are computed using shift(1) in
    build_raw_feature_frame, so they already use only data from T-1 and earlier.
    We do NOT recompute them here to preserve correct semantics.
    """
    if train_data.empty:
        return test_row
    test_row = test_row.copy()
    test_row_date = test_row["date"].iloc[0]
    # Last known DA and its date (from training data only)
    last_da = float(train_data["da_raw"].iloc[-1])
    last_obs_date = train_data["date"].iloc[-1]
    test_row["last_observed_da_raw"] = last_da
    test_row["weeks_since_last_raw"] = (test_row_date - last_obs_date).days / 7.0
    # Last spike date from training only
    spike_mask = train_data["da_raw"] > spike_threshold
    if spike_mask.any():
        last_spike_date = train_data.loc[spike_mask, "date"].max()
        test_row["weeks_since_last_spike"] = (test_row_date - last_spike_date).days / 7.0
    else:
        test_row["weeks_since_last_spike"] = 999.0
    # NOTE: Do NOT recompute rolling features - they already use shift(1) and don't leak
    return test_row


def get_site_test_row(
    feature_frame: pd.DataFrame,
    site: str,
    test_date: pd.Timestamp,
    anchor_date: pd.Timestamp,
    max_date_diff_days: int = 14,
) -> Optional[pd.DataFrame]:
    """
    LEGACY: Find closest processed row to test_date (after anchor_date).

    WARNING: This uses environmental features from the test date, which
    constitutes data leakage for a forecasting task.  Use
    get_site_anchor_row() instead for leak-free validation.
    """
    test_date = pd.Timestamp(test_date)
    anchor_date = pd.Timestamp(anchor_date)
    site_data = feature_frame[feature_frame["site"] == site].copy()
    site_data = site_data.sort_values("date")
    future_data = site_data[site_data["date"] > anchor_date].copy()
    if future_data.empty:
        return None
    future_data["date_diff"] = abs((future_data["date"] - test_date).dt.days)
    closest_idx = future_data["date_diff"].idxmin()
    if future_data.loc[closest_idx, "date_diff"] > max_date_diff_days:
        return None
    return future_data.loc[[closest_idx]].drop(columns=["date_diff"])


def get_site_anchor_row(
    feature_frame: pd.DataFrame,
    site: str,
    test_date: pd.Timestamp,
    anchor_date: pd.Timestamp,
    max_date_diff_days: int = 28,
) -> Optional[pd.DataFrame]:
    """
    Build a leak-free test row for forecasting validation.

    Environmental features (SST, chlorophyll, BEUTI, etc.) come from the
    closest processed row AT or BEFORE anchor_date — information that would
    actually be available at prediction time.

    The row's ``date`` is set to ``test_date`` so that deterministic calendar
    features (sin_day_of_year, month, ...) are computed correctly downstream.
    Persistence / lag features should still be overwritten via
    ``recompute_test_row_persistence_features()`` after calling this.
    """
    test_date = pd.Timestamp(test_date)
    anchor_date = pd.Timestamp(anchor_date)
    site_data = feature_frame[feature_frame["site"] == site].copy()
    site_data = site_data.sort_values("date")

    # Environmental features from anchor date or earlier
    past_data = site_data[site_data["date"] <= anchor_date].copy()
    if past_data.empty:
        return None

    # Most recent available env data (closest to anchor)
    past_data["date_diff"] = abs((past_data["date"] - anchor_date).dt.days)
    closest_idx = past_data["date_diff"].idxmin()

    if past_data.loc[closest_idx, "date_diff"] > max_date_diff_days:
        return None

    anchor_row = past_data.loc[[closest_idx]].drop(columns=["date_diff"]).copy()

    # Set date to test_date so temporal features will be computed for the
    # actual prediction date (these are deterministic calendar features,
    # not leaked observations).
    anchor_row["date"] = test_date
    anchor_row["site"] = site

    return anchor_row


def get_last_known_raw_da(
    train_data: pd.DataFrame,
) -> Optional[float]:
    if train_data is None or train_data.empty:
        return None
    return float(train_data["da_raw"].iloc[-1])

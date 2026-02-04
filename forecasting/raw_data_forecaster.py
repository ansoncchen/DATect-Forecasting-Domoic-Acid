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
from .data_processor import DataProcessor
from .torch_forecasting_adapter import build_timeseries_dataset
from .models.tft_model import TFTRegressor
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RawForecastConfig:
    lags: Iterable[int] = (1, 2, 3, 52)
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

    processor = DataProcessor()
    merged = processor.create_raw_lag_features(
        merged, group_col="site", value_col="da_raw", lags=list(cfg.lags)
    )
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


def get_site_test_row(
    feature_frame: pd.DataFrame,
    site: str,
    test_date: pd.Timestamp,
    anchor_date: pd.Timestamp,
    max_date_diff_days: int = 14,
) -> Optional[pd.DataFrame]:
    """
    Find closest processed row to test_date (after anchor_date).
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


def get_last_known_raw_da(
    train_data: pd.DataFrame,
) -> Optional[float]:
    if train_data is None or train_data.empty:
        return None
    return float(train_data["da_raw"].iloc[-1])


def build_tft_dataset_for_raw(
    feature_frame: pd.DataFrame,
    max_encoder_length: int = 26,
    max_prediction_length: int = 1,
):
    """
    Build a TimeSeriesDataSet for TFT training using raw DA measurements.
    """
    if "da_raw" not in feature_frame.columns:
        raise ValueError("da_raw column is required for TFT dataset.")
    dataset, _, _ = build_timeseries_dataset(
        feature_frame,
        target_col="da_raw",
        group_col="site",
        time_col="date",
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
    )
    return dataset


def fit_tft_on_raw(
    feature_frame: pd.DataFrame,
    max_encoder_length: int = 26,
    max_prediction_length: int = 1,
    tft_params: Optional[dict] = None,
):
    """
    Fit a TFT model on raw DA data using the adapter utilities.
    """
    dataset = build_tft_dataset_for_raw(
        feature_frame,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
    )
    model = TFTRegressor(**(tft_params or {}))
    model.fit(dataset)
    return model, dataset

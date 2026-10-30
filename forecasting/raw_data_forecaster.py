"""
Raw Data Forecasting Utilities
==============================

Builds training datasets from environmental features and DA measurements.
When ``USE_INTERPOLATED_TRAINING`` is True (default), training includes both
real raw DA measurements and gap-filled (interpolated) values for ~5x more
data.  Test/evaluation always uses real raw measurements only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import os
import pandas as pd
import numpy as np

import config
from .measurements import parse_da_values, parse_dates
from .raw_data_processor import RawDataProcessor as DataProcessor
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RawForecastConfig:
    lags: Iterable[int] = None  # Read from config at runtime, not import time

    def __post_init__(self):
        if self.lags is None:
            self.lags = tuple(config.LAG_FEATURES)


def _normalize_site_name(site_key: str) -> str:
    return site_key.replace("-da", "").replace("_da", "").replace("-", " ").replace("_", " ").title()


def load_raw_da_measurements() -> pd.DataFrame:
    """
    Load raw DA measurements from CSVs.

    Returns: DataFrame with columns [date, site, da_raw], one row per site-day.
    Several samples on the same site-day (the CSVs record no species or station)
    collapse to their maximum, matching the weekly MAX aggregation used for
    training targets here and in dataset-creation.py.
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

        df["date"] = parse_dates(df[date_col])
        df["da_raw"] = parse_da_values(df[da_col])  # "<1" -> LOD/2
        df["site"] = site_name

        valid_df = df.dropna(subset=["date", "da_raw"])
        valid_df = valid_df[valid_df["da_raw"] >= 0]
        raw_measurements.append(valid_df[["date", "site", "da_raw"]])

    if not raw_measurements:
        raise ValueError("No raw DA measurements could be loaded.")

    all_raw = pd.concat(raw_measurements, ignore_index=True)
    all_raw = all_raw.groupby(["site", "date"], as_index=False)["da_raw"].max()
    return all_raw[["date", "site", "da_raw"]]


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
) -> pd.DataFrame:
    """
    Merge raw DA (weekly) into processed environmental features and add raw lags.
    """
    cfg = RawForecastConfig()
    base = processed_data.copy()
    base["date"] = pd.to_datetime(base["date"])

    raw_weekly = raw_weekly.copy()
    raw_weekly["date"] = pd.to_datetime(raw_weekly["date"])

    merged = base.merge(raw_weekly, on=["site", "date"], how="left")

    # --- Derived environmental features (no leakage: env columns only) ---
    merged = merged.sort_values(["site", "date"]).reset_index(drop=True)

    # Pseudo-nitzschia log transform: compresses heavy-tailed distribution.
    # pn_above_threshold was also tried but confirmed negligible (ΔR²=-0.005).
    if "pn" in merged.columns:
        merged["pn_log"] = np.log1p(merged["pn"])

    # Note: Several derived features were explored during development
    # (pdo_oni_phase, mhw_flag, beuti_squared, beuti_relaxation,
    # fluor_efficiency, k490_squared, pn_above_threshold) but individual
    # ablation confirmed all have |ΔR²| ≤ 0.005. They have been removed
    # from the pipeline. See paper_feature_ablation_results.json.

    # Persistence features from raw measurements (no interpolation), using only
    # observations before the row's own week. Including the row itself made the
    # feature equal the training target (and weeks_since_last_spike 0 exactly on
    # spike rows), which never happens at forecast time.
    merged = merged.sort_values(["site", "date"])
    previous_da = merged.groupby("site")["da_raw"].shift(1)
    merged["last_observed_da_raw"] = previous_da.groupby(merged["site"]).ffill()

    spike_date = merged["date"].where(merged["da_raw"] >= config.SPIKE_THRESHOLD)
    last_spike_date = spike_date.groupby(merged["site"]).shift(1)
    last_spike_date = last_spike_date.groupby(merged["site"]).ffill()
    merged["weeks_since_last_spike"] = (merged["date"] - last_spike_date).dt.days / 7.0
    merged["weeks_since_last_spike"] = merged["weeks_since_last_spike"].fillna(999.0)

    if config.USE_ROLLING_FEATURES:
        rolling_windows = (4, 8, 12)
        # Already excludes the row's own value (same series the old shift(1) produced)
        shifted = merged["last_observed_da_raw"]
        for window in rolling_windows:
            rolling_group = shifted.groupby(merged["site"]).rolling(window, min_periods=1)
            # Rolling mean only needed for window=4 (8/12 are negligible)
            if window == 4:
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


# A weekly row is labelled with its Monday but aggregates Monday-Sunday, so it
# can hold measurements up to six days after its date. A row may be used at an
# anchor only once its whole week has passed.
WEEK_END_OFFSET = pd.Timedelta(days=6)


def week_complete_by(dates: pd.Series, anchor_date) -> pd.Series:
    """True where the weekly row dated ``dates`` ends on or before ``anchor_date``."""
    return dates + WEEK_END_OFFSET <= pd.Timestamp(anchor_date)


def get_site_training_frame(
    feature_frame: pd.DataFrame,
    site: str,
    anchor_date: pd.Timestamp,
    min_training_samples: int = 10,
) -> Optional[pd.DataFrame]:
    """
    Select training rows for a site.

    When USE_INTERPOLATED_TRAINING is True, includes all rows that have
    either real or gap-filled DA values (~5x more training data).
    Real measurements are always preferred; interpolated values fill gaps.
    A ``_is_interpolated`` marker column is added so downstream code can
    filter to real-only rows for persistence feature recomputation.

    Only weeks that end by ``anchor_date`` are used: the anchor's own week can
    contain measurements taken after the anchor.
    """
    anchor_date = pd.Timestamp(anchor_date)
    site_data = feature_frame[feature_frame["site"] == site].copy()
    site_data = site_data.sort_values("date")
    train_data = site_data[week_complete_by(site_data["date"], anchor_date)].copy()

    if config.USE_INTERPOLATED_TRAINING:
        # Keep rows that have either real or interpolated DA
        train_data = train_data.dropna(subset=["da"])
        # Mark which rows are interpolated (before we fill da_raw)
        train_data["_is_interpolated"] = train_data["da_raw"].isna()
        # Training target: prefer real measurement, fall back to interpolated
        train_data["da_raw"] = train_data["da_raw"].fillna(train_data["da"])
    else:
        train_data = train_data.dropna(subset=["da_raw"])
        train_data["_is_interpolated"] = False

    if len(train_data) < min_training_samples:
        return None
    return train_data


def recompute_test_row_persistence_features(
    test_row: pd.DataFrame,
    observations: pd.DataFrame,
    spike_threshold: float,
) -> pd.DataFrame:
    """
    Overwrite persistence features in test_row from ``observations``: real DA
    measurements (columns ``date``, ``da_raw``) dated on or before the anchor,
    sorted by date. The engine passes daily raw measurements so a sample taken
    earlier in the anchor's week still counts and one taken after it does not.

    Only these features can leak the target:
    - last_observed_da_raw: the anchor row's value comes from its weekly bucket
    - weeks_since_last_spike: likewise

    NOTE: Rolling features (raw_obs_roll_*) exclude each row's own week in
    build_raw_feature_frame, and the anchor row's week ends by the anchor, so
    they are not recomputed here.
    """
    if observations.empty:
        return test_row
    test_row = test_row.copy()
    test_row_date = test_row["date"].iloc[0]
    last_da = float(observations["da_raw"].iloc[-1])
    test_row["last_observed_da_raw"] = last_da
    spike_mask = observations["da_raw"] >= spike_threshold
    if spike_mask.any():
        last_spike_date = observations.loc[spike_mask, "date"].max()
        test_row["weeks_since_last_spike"] = (test_row_date - last_spike_date).days / 7.0
    else:
        test_row["weeks_since_last_spike"] = 999.0
    # NOTE: Do NOT recompute rolling features - they already use shift(1) and don't leak
    return test_row


def get_site_anchor_row(
    feature_frame: pd.DataFrame,
    site: str,
    test_date: pd.Timestamp,
    anchor_date: pd.Timestamp,
    max_date_diff_days: int = 28,
) -> Optional[pd.DataFrame]:
    """
    Build a leak-free test row for forecasting validation.

    Environmental features (SST, chlorophyll, BEUTI, PN, etc.) come from the
    latest processed row whose week ends by anchor_date — information that would
    actually be available at prediction time (a weekly PN maximum labelled with
    the anchor's Monday can include samples taken after the anchor).

    The row's ``date`` is set to ``test_date`` so that deterministic calendar
    features (sin_day_of_year, month, ...) are computed correctly downstream.
    Persistence / lag features should still be overwritten via
    ``recompute_test_row_persistence_features()`` after calling this.
    """
    test_date = pd.Timestamp(test_date)
    anchor_date = pd.Timestamp(anchor_date)
    site_data = feature_frame[feature_frame["site"] == site].copy()
    site_data = site_data.sort_values("date")

    # Environmental features from weeks that ended by the anchor
    past_data = site_data[week_complete_by(site_data["date"], anchor_date)].copy()
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
    anchor_date: Optional[pd.Timestamp] = None,
    max_age_days: Optional[int] = None,
) -> Optional[float]:
    if train_data is None or train_data.empty:
        return None
    df = train_data.copy()
    if "date" in df.columns:
        df = df.sort_values("date")
        if anchor_date is not None:
            anchor_date = pd.Timestamp(anchor_date)
            df = df[df["date"] <= anchor_date]
    if df.empty:
        return None
    if max_age_days is not None and "date" in df.columns and anchor_date is not None:
        last_date = df["date"].iloc[-1]
        age_days = (anchor_date - last_date).days
        if age_days > max_age_days:
            return None
    return float(df["da_raw"].iloc[-1])

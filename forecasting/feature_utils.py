"""
Feature Engineering Utilities
=============================

Shared helpers for temporal feature creation and data preprocessing.
Extracted from the raw-data validation pipeline to avoid duplication
between RawForecastEngine, validation scripts, and future modules.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import config


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add deterministic temporal features matching the main pipeline.

    These features are safe to compute for any date because they only
    depend on the calendar (day-of-year, month, etc.) — not on
    observed data that could leak future information.

    Features added (when USE_ENHANCED_TEMPORAL_FEATURES=True):
      sin_day_of_year, cos_day_of_year, month, sin_month, cos_month,
      quarter, sin_week_of_year, cos_week_of_year, is_bloom_season,
      days_since_start

    Ported from validate_on_raw_data.py lines 198-219.
    """
    df = df.copy()

    if getattr(config, "USE_ENHANCED_TEMPORAL_FEATURES", True):
        day_of_year = df["date"].dt.dayofyear
        df["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365)
        df["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365)
        df["month"] = df["date"].dt.month
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
        df["quarter"] = df["date"].dt.quarter
        week_of_year = df["date"].dt.isocalendar().week.astype(int)
        df["sin_week_of_year"] = np.sin(2 * np.pi * week_of_year / 52)
        df["cos_week_of_year"] = np.cos(2 * np.pi * week_of_year / 52)
        df["is_bloom_season"] = df["month"].between(3, 10).astype(int)
        df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
    else:
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter

    return df


def create_transformer(
    df: pd.DataFrame,
    drop_cols: list[str],
    include_cols: Optional[list[str]] = None,
) -> tuple[ColumnTransformer, pd.DataFrame]:
    """
    Create an Impute → MinMaxScale preprocessing transformer.

    The transformer operates on numeric columns only, dropping everything
    else (strings, dates, etc.).  Columns listed in *drop_cols* are
    removed from the feature matrix before the transformer is built.

    Returns:
        (transformer, X)  where X = df.drop(drop_cols).select(numeric)

    Ported from validate_on_raw_data.py lines 222-244.
    """
    X = df.drop(columns=drop_cols, errors="ignore")
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if include_cols:
        include_set = set(include_cols)
        numeric_cols = [c for c in numeric_cols if c in include_set]
    # Drop columns that are entirely NaN so median imputation succeeds
    numeric_cols = [c for c in numeric_cols if X[c].notna().any()]

    if len(numeric_cols) == 0:
        raise ValueError("No numeric features available after dropping columns")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ])

    transformer = ColumnTransformer(
        [("num", numeric_pipeline, numeric_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    transformer.set_output(transform="pandas")

    return transformer, X

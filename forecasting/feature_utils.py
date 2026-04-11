"""
Feature Engineering Utilities
=============================

Shared helpers for temporal feature creation and data preprocessing.
Extracted to avoid duplication across RawForecastEngine and related code.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add deterministic temporal features matching the main pipeline.

    These features are safe to compute for any date because they only
    depend on the calendar (day-of-year, month, etc.) — not on
    observed data that could leak future information.

    Features added:
      sin_day_of_year, cos_day_of_year, month
    """
    df = df.copy()

    day_of_year = df["date"].dt.dayofyear
    df["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365)
    df["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365)
    df["month"] = df["date"].dt.month

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

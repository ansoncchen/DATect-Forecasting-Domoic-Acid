"""
PyTorch Forecasting adapter utilities.

Converts DATect's weekly site data into TimeSeriesDataSet format.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from pytorch_forecasting import TimeSeriesDataSet
    HAS_PF = True
except ImportError:
    TimeSeriesDataSet = None
    HAS_PF = False


KNOWN_REAL_CANDIDATES = [
    "sin_day_of_year",
    "cos_day_of_year",
    "month",
    "sin_month",
    "cos_month",
    "quarter",
    "days_since_start",
]


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def prepare_timeseries_dataframe(
    df: pd.DataFrame,
    group_col: str = "site",
    time_col: str = "date",
) -> pd.DataFrame:
    """
    Add time_idx per site and ensure sorted input for TimeSeriesDataSet.
    """
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.sort_values([group_col, time_col]).reset_index(drop=True)
    out["time_idx"] = out.groupby(group_col).cumcount().astype(int)
    return out


def build_timeseries_dataset(
    df: pd.DataFrame,
    target_col: str = "da",
    group_col: str = "site",
    time_col: str = "date",
    max_encoder_length: int = 26,
    max_prediction_length: int = 1,
    fill_strategy: str = "median",
) -> Tuple["TimeSeriesDataSet", List[str], List[str]]:
    """
    Build a TimeSeriesDataSet and return it along with known/unknown real feature lists.
    """
    if not HAS_PF:
        raise ImportError("pytorch-forecasting is required for TimeSeriesDataSet.")

    data = prepare_timeseries_dataframe(df, group_col=group_col, time_col=time_col)

    known_reals = [c for c in KNOWN_REAL_CANDIDATES if c in data.columns]

    numeric_cols = _numeric_columns(data)
    excluded = {target_col, "time_idx"}
    excluded.update(known_reals)
    excluded.add(time_col)
    excluded.add(group_col)

    unknown_reals = [c for c in numeric_cols if c not in excluded]

    # Fill missing values for numeric columns
    if fill_strategy == "median":
        for col in numeric_cols:
            median = data[col].median()
            data[col] = data[col].fillna(median)
    else:
        data[numeric_cols] = data[numeric_cols].fillna(0.0)

    dataset = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target=target_col,
        group_ids=[group_col],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        static_categoricals=[group_col],
    )

    return dataset, known_reals, unknown_reals


def make_dataloaders(dataset: "TimeSeriesDataSet", batch_size: int = 64):
    """
    Create train/val dataloaders from a TimeSeriesDataSet.
    """
    train_loader = dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    return train_loader, val_loader

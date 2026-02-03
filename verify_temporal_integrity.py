#!/usr/bin/env python3
"""
Temporal integrity validation script.
"""

import sys
import pandas as pd

import config
from forecasting.data_processor import DataProcessor
from forecasting.forecast_engine import ForecastEngine


def validate_site_ordering(df):
    for site in df["site"].unique():
        site_df = df[df["site"] == site].sort_values("date")
        if not site_df["date"].is_monotonic_increasing:
            raise ValueError(f"Date ordering violated for site: {site}")


def validate_lag_masking(df, anchor_date):
    processor = DataProcessor()
    lags = config.LAG_FEATURES
    if not lags:
        return
    lagged = processor.create_lag_features_safe(df, "site", "da", lags, anchor_date)
    cutoff_date = anchor_date - pd.Timedelta(days=1)
    future_mask = lagged["date"] > cutoff_date
    for lag in lags:
        col = f"da_lag_{lag}"
        if col in lagged.columns and lagged.loc[future_mask, col].notna().any():
            raise ValueError(f"Lag masking failed for {col} after cutoff {cutoff_date}")


def run_smoke_retrospective():
    engine = ForecastEngine()
    results = engine.run_retrospective_evaluation(
        task="regression",
        model_type="xgboost",
        n_anchors=10,
        min_test_date="2008-01-01"
    )
    if results is None or results.empty:
        raise ValueError("Retrospective evaluation produced no results")


def main():
    processor = DataProcessor()
    data = processor.load_and_prepare_base_data(config.FINAL_OUTPUT_PATH)

    validate_site_ordering(data)

    anchor_date = data["date"].quantile(0.6)
    validate_lag_masking(data, anchor_date)

    run_smoke_retrospective()

    print("Temporal integrity validation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

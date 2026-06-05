"""
Unit tests for backend/visualizations.py math robustness and edge cases.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.visualizations import (
    generate_gradient_uncertainty_plot,
    generate_correlation_heatmap,
    generate_sensitivity_analysis,
    generate_time_series_comparison,
    generate_waterfall_plot,
    generate_spectral_analysis,
)


def test_gradient_uncertainty_plot_normal():
    """Verify normal quantile input generates a valid Plotly JSON string."""
    quantiles = {"q05": 5.0, "q50": 12.0, "q95": 25.0}
    json_str = generate_gradient_uncertainty_plot(
        quantiles, ensemble_prediction=13.5, xgb_prediction=14.0, rf_prediction=13.0
    )
    assert isinstance(json_str, str)
    assert "Ensemble Median" in json_str
    assert "XGBoost" in json_str
    assert "Random Forest" in json_str


def test_gradient_uncertainty_plot_collapsed():
    """Verify collapsed quantiles (zero width/identical values) handle divisions safely without crashing."""
    quantiles = {"q05": 10.0, "q50": 10.0, "q95": 10.0}
    json_str = generate_gradient_uncertainty_plot(
        quantiles, ensemble_prediction=10.0, xgb_prediction=10.0, rf_prediction=10.0
    )
    assert isinstance(json_str, str)
    assert "Ensemble Median" in json_str


def test_correlation_heatmap_nan_columns():
    """Verify correlation heatmap handles all-NaN columns or constant values gracefully."""
    df = pd.DataFrame({
        "site": ["Kalaloch"] * 5,
        "date": pd.date_range("2023-01-01", periods=5, freq="W"),
        "da": [1.0, 2.0, 3.0, np.nan, 5.0],
        "all_nan_col": [np.nan] * 5,
        "constant_col": [42.0] * 5
    })

    plot_data = generate_correlation_heatmap(df, site="Kalaloch")
    assert "data" in plot_data
    assert "layout" in plot_data
    # Should compile without division by zero or NaN crashes


def test_sensitivity_analysis_tiny_data():
    """Verify sensitivity analysis handles tiny datasets safely (returning default plots or correlation-only)."""
    df = pd.DataFrame({
        "site": ["Copalis"] * 3,
        "date": pd.date_range("2023-01-01", periods=3, freq="W"),
        "da": [2.0, 4.0, 6.0],
        "modis-sst": [12.0, 13.0, 14.0],
        "pdo": [0.5, 0.6, 0.7]
    })

    plots = generate_sensitivity_analysis(df, site="Copalis")
    assert len(plots) >= 1
    assert "data" in plots[0]


def test_time_series_comparison():
    """Verify time series comparison handles normal and missing columns cleanly."""
    df = pd.DataFrame({
        "site": ["Copalis"] * 5,
        "date": pd.date_range("2023-01-01", periods=5, freq="W"),
        "da": [5.0, 10.0, 15.0, 20.0, 25.0],
        "pn": [100.0, 200.0, 300.0, 400.0, 500.0]
    })

    plot_data = generate_time_series_comparison(df, site="Copalis")
    assert "data" in plot_data
    assert len(plot_data["data"]) == 2  # DA and PN


def test_waterfall_plot():
    """Verify waterfall plot handles site grouping and latitude baselines properly."""
    df = pd.DataFrame({
        "site": ["Copalis", "Kalaloch", "Copalis", "Kalaloch"],
        "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-08", "2023-01-08"]),
        "da": [2.0, 5.0, 8.0, 12.0],
        "lat": [47.1, 47.5, 47.1, 47.5]
    })

    plot_data = generate_waterfall_plot(df)
    assert "data" in plot_data
    assert len(plot_data["data"]) >= 2


def test_spectral_analysis_short():
    """Verify spectral analysis returns an empty list or handles short data cleanly without crash."""
    df = pd.DataFrame({
        "site": ["Copalis"] * 5,
        "date": pd.date_range("2023-01-01", periods=5, freq="W"),
        "da": [1.0, 2.0, 3.0, 4.0, 5.0]
    })

    plots = generate_spectral_analysis(df, site="Copalis")
    assert plots == []  # returns empty list for len < 20

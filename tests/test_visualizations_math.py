"""
Unit tests for backend/visualizations.py math robustness and edge cases.
"""

from pathlib import Path
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent

from backend.visualizations import (
    generate_band_plot,
    generate_correlation_heatmap,
    generate_sensitivity_analysis,
    generate_time_series_comparison,
    generate_waterfall_plot,
    generate_spectral_analysis,
)


def test_band_plot_draws_range_forecast_and_threshold():
    """The forecast chart shows the 80% conformal range, the forecast and the spike line, in ug/g."""
    import json
    band = {"lo": 4.2, "hi": 31.5, "level": 0.8, "n_calibration": 500}
    fig = json.loads(generate_band_plot(band, 13.5, xgb_prediction=14.0, rf_prediction=13.0))
    traces = {t["name"]: t for t in fig["data"]}
    assert list(traces["80% range (past errors)"]["x"]) == [4.2, 31.5]
    assert list(traces["Forecast"]["x"]) == [13.5]
    assert "XGBoost" in traces and "Random Forest" in traces
    assert fig["layout"]["xaxis"]["title"]["text"] == "DA (µg/g)"
    assert any(sh["x0"] == 20 and sh["x1"] == 20 for sh in fig["layout"]["shapes"])
    raw = generate_band_plot(band, 13.5)
    assert "μg/L" not in raw and "µg/L" not in raw


def test_band_plot_with_a_collapsed_range():
    """A zero-width range (every past error the same) still renders."""
    import json
    fig = json.loads(generate_band_plot({"lo": 10.0, "hi": 10.0, "level": 0.8}, 10.0))
    assert {t["name"] for t in fig["data"]} == {"80% range (past errors)", "Forecast"}


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


def test_weekly_figures_are_labelled_gap_filled():
    """Charts drawn from the gap-filled weekly series say so in their title."""
    df = pd.DataFrame({
        "site": ["Copalis", "Kalaloch"] * 3,
        "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-08", "2023-01-08", "2023-01-15", "2023-01-15"]),
        "da": [2.0, 5.0, 8.0, 12.0, 3.0, 4.0],
        "pn": [100.0, 200.0, 300.0, 400.0, 150.0, 250.0],
        "lat": [47.1, 47.5] * 3,
    })
    comparison = generate_time_series_comparison(df, site="Copalis")["layout"]["title"]
    waterfall = generate_waterfall_plot(df)["layout"]["title"]
    for title in (comparison, waterfall):
        assert "gaps filled" in (title["text"] if isinstance(title, dict) else title)

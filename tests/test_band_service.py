"""The dashboard's error band, computed when a forecast is served, must equal the
band the declared protocol evaluated (docs/ERROR_BAND.md) for the same forecast."""
import pandas as pd
import pytest

from backend import band


def test_serve_time_band_equals_the_evaluated_band():
    rows = pd.read_parquet("results/conformal_band/band_predictions.parquet").dropna(subset=["band_lo"])
    for _, r in rows.sample(50, random_state=0).iterrows():
        b = band.forecast_band(r.predicted_da, r.anchor_date)
        assert (b["lo"], b["hi"], b["p_exceed"]) == (r.band_lo, r.band_hi, r.p_exceed)  # exact
        assert b["n_calibration"] == r.n_calibration and b["level"] == 0.8


def test_no_band_before_thirty_errors_or_for_nan():
    assert band.forecast_band(5.0, pd.Timestamp("2005-01-01")) is None
    assert band.forecast_band(float("nan"), pd.Timestamp("2020-01-01")) is None


def test_category_probabilities_sum_to_one_and_respect_bins():
    p = band.forecast_band(10.0, pd.Timestamp("2021-06-01"))["category_probabilities"]
    assert len(p) == 4 and sum(p) == pytest.approx(1.0)


def test_chosen_variant_comes_from_the_declared_selection():
    assert band.chosen_variant() == {"coverage": 0.8, "window": 500}

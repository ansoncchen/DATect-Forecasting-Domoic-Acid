"""The conformal band must use only errors known by each forecast's anchor and
reach its nominal coverage when errors are exchangeable."""
import numpy as np
import pandas as pd
import pytest

from forecasting.conformal import (
    band_from_residuals,
    exceedance_probability,
    interval_score,
    log_residuals,
    past_residual_bands,
)


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2010-01-04") + pd.to_timedelta(7 * np.arange(n), unit="D")
    pred = rng.gamma(1.5, 4.0, n)
    actual = np.expm1(np.log1p(pred) + rng.normal(0, 0.6, n)).clip(0)
    return pd.DataFrame({"date": dates, "anchor_date": dates - pd.Timedelta(days=7),
                         "actual_da": actual, "predicted_da": pred})


def test_band_uses_finite_sample_ranks():
    resid = np.arange(-4, 5) / 10.0  # n = 9: ranks floor(10*0.1)=1 and ceil(10*0.9)=9
    lo, hi = band_from_residuals(10.0, resid, coverage=0.8)
    assert lo == pytest.approx(np.expm1(np.log1p(10.0) - 0.4))
    assert hi == pytest.approx(np.expm1(np.log1p(10.0) + 0.4))
    assert band_from_residuals(10.0, np.zeros(30), 0.8) == pytest.approx((10.0, 10.0))


def test_band_lower_edge_never_goes_below_zero():
    lo, _ = band_from_residuals(0.0, np.full(30, -2.0), 0.8)
    assert lo == 0.0


def test_exceedance_probability_counts_errors_that_reach_the_threshold():
    resid = log_residuals([20, 5, 40, 0], [10, 10, 10, 10])  # what would carry 10 to 20?
    assert exceedance_probability(10.0, resid, 20.0) == 0.5  # 20 counts, 40 counts


def test_band_ignores_measurements_after_the_anchor():
    frame = _frame()
    base = past_residual_bands(frame, window=100)
    i = 250
    future = pd.to_datetime(frame["date"]) > frame.loc[i, "anchor_date"]
    changed = frame.copy()
    changed.loc[future, "actual_da"] = 500.0  # rewrite everything after row i's anchor
    pd.testing.assert_series_equal(past_residual_bands(changed, window=100).loc[i], base.loc[i])
    assert base.loc[i, "n_calibration"] == 100
    # Weekly rows: row i's anchor is row i-1's date, so rows 0..i-1 count (the anchor is inclusive)
    assert past_residual_bands(frame).loc[i, "n_calibration"] == i


def test_rows_without_enough_history_get_no_band():
    out = past_residual_bands(_frame(n=40), min_calibration=30)
    assert out["band_lo"].iloc[:30].isna().all() and out["band_lo"].iloc[30:].notna().all()


@pytest.mark.parametrize("mondrian", [False, True])
def test_exchangeable_errors_reach_nominal_coverage(mondrian):
    frame = _frame(n=3000, seed=1)
    out = past_residual_bands(frame, coverage=0.8, mondrian=mondrian).dropna()
    y = frame.loc[out.index, "actual_da"]
    covered = (y >= out["band_lo"]) & (y <= out["band_hi"])
    assert abs(covered.mean() - 0.8) < 0.025


def test_interval_score_is_the_width_when_the_measurement_is_inside():
    assert interval_score([5.0], [2.0], [9.0], 0.8)[0] == pytest.approx(np.log1p(9) - np.log1p(2))
    # outside by d on the log scale adds 2/alpha * d
    assert interval_score([0.0], [2.0], [9.0], 0.8)[0] == pytest.approx(
        np.log1p(9) - np.log1p(2) + 10 * np.log1p(2))

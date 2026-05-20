"""Unit tests for forecasting.oad_features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.oad_features import (
    LEAK_SHIFT_DAYS,
    OAD_FEATURES_ALL,
    add_oad_features,
    compute_region_features,
)


@pytest.fixture
def synthetic_oad():
    dates = pd.date_range("2010-01-01", "2014-12-31", freq="D")
    base = np.arange(len(dates)) * 0.01
    seasonal = 0.5 * np.sin(2 * np.pi * dates.dayofyear / 365)
    return pd.DataFrame({"date": dates, "score": base + seasonal})


def test_oad_score_is_7day_mean_ending_at_row_minus_5(synthetic_oad):
    feats = compute_region_features(synthetic_oad, pd.to_datetime(["2013-06-10"]))
    R = pd.Timestamp("2013-06-10")
    s = synthetic_oad.set_index("date")["score"]
    expected = s.loc[R - pd.Timedelta(days=11) : R - pd.Timedelta(days=5)].mean()
    assert feats.loc[0, "oad_score"] == pytest.approx(expected)


def test_lag_features_are_7day_means_one_and_two_weeks_earlier(synthetic_oad):
    feats = compute_region_features(synthetic_oad, pd.to_datetime(["2013-06-10"]))
    R = pd.Timestamp("2013-06-10")
    s = synthetic_oad.set_index("date")["score"]
    expected_lag1 = s.loc[R - pd.Timedelta(days=18) : R - pd.Timedelta(days=12)].mean()
    expected_lag2 = s.loc[R - pd.Timedelta(days=25) : R - pd.Timedelta(days=19)].mean()
    assert feats.loc[0, "oad_score_lag1week"] == pytest.approx(expected_lag1)
    assert feats.loc[0, "oad_score_lag2week"] == pytest.approx(expected_lag2)


def test_30day_window_ends_at_row_minus_5(synthetic_oad):
    feats = compute_region_features(synthetic_oad, pd.to_datetime(["2013-06-10"]))
    R = pd.Timestamp("2013-06-10")
    window = synthetic_oad.set_index("date").loc[
        R - pd.Timedelta(days=35) : R - pd.Timedelta(days=5), "score"
    ]
    assert feats.loc[0, "oad_score_30day_mean"] == pytest.approx(window.mean())
    assert feats.loc[0, "oad_score_30day_max"] == pytest.approx(window.max())


def test_30day_trend_is_linear_slope(synthetic_oad):
    """Base series rises 0.01/day; near June peak the sine flattens it but slope stays positive."""
    feats = compute_region_features(synthetic_oad, pd.to_datetime(["2013-06-10"]))
    # Just require positive trend in the right order of magnitude (sine dampens it)
    assert 0.001 < feats.loc[0, "oad_score_30day_trend"] < 0.02


def test_doy_zscore_uses_only_prior_years(synthetic_oad):
    feats = compute_region_features(synthetic_oad, pd.to_datetime(["2010-06-10"]))
    assert pd.isna(feats.loc[0, "oad_score_zscore_doy"]), \
        "DOY z-score in first year must be NaN (no climatology baseline)"
    feats = compute_region_features(synthetic_oad, pd.to_datetime(["2013-06-10"]))
    assert not pd.isna(feats.loc[0, "oad_score_zscore_doy"])


def test_missing_oad_dates_yield_nan(synthetic_oad):
    feats = compute_region_features(synthetic_oad, pd.to_datetime(["2020-06-10"]))
    assert pd.isna(feats.loc[0, "oad_score"])
    assert pd.isna(feats.loc[0, "oad_score_lag1week"])


def test_leak_invariant(synthetic_oad):
    """Zeroing scores after R-5 must not change any feature value."""
    R = pd.Timestamp("2013-06-10")
    cutoff = R - pd.Timedelta(days=LEAK_SHIFT_DAYS)
    feats_full = compute_region_features(synthetic_oad, pd.DatetimeIndex([R]))

    truncated = synthetic_oad.copy()
    truncated.loc[truncated["date"] > cutoff, "score"] = np.nan
    feats_trunc = compute_region_features(truncated, pd.DatetimeIndex([R]))

    for col in feats_full.columns:
        a, b = feats_full.loc[0, col], feats_trunc.loc[0, col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == pytest.approx(b), \
            f"Feature {col} depends on OAD data after R-5 ({cutoff.date()}) — LEAK"


def test_add_oad_features_adds_16_columns(synthetic_oad, tmp_path):
    rows = []
    for region in ["SW Washington / Long Beach", "Overall (WA–OR–N. CA coastal)"]:
        rows.append(synthetic_oad.assign(region=region, method="ae3d", aggregation="mean"))
    pd.concat(rows, ignore_index=True).to_parquet(tmp_path / "oad.parquet")

    compiled = pd.DataFrame({
        "Date": pd.to_datetime(["2013-06-03", "2013-06-10", "2013-06-17"]),
        "Site": ["Twin Harbors"] * 3,
    })
    out = add_oad_features(compiled, str(tmp_path / "oad.parquet"))
    for col in OAD_FEATURES_ALL:
        assert col in out.columns, f"missing {col}"
    assert len(out) == 3
    assert out["oad_valid_frac"].isna().all()
    assert out["oad_overall_valid_frac"].isna().all()


def test_add_oad_features_sites_in_different_regions_get_different_scores(synthetic_oad, tmp_path):
    def make_region(region, val):
        return pd.DataFrame({
            "date": synthetic_oad["date"],
            "score": [val] * len(synthetic_oad),
            "region": region, "method": "ae3d", "aggregation": "mean",
        })
    pd.concat([
        make_region("Olympic Coast (WA)", 1.0),
        make_region("SW Washington / Long Beach", 5.0),
        make_region("Overall (WA–OR–N. CA coastal)", 3.0),
    ], ignore_index=True).to_parquet(tmp_path / "oad.parquet")

    compiled = pd.DataFrame({
        "Date": pd.to_datetime(["2013-06-10", "2013-06-10"]),
        "Site": ["Kalaloch", "Twin Harbors"],
    })
    out = add_oad_features(compiled, str(tmp_path / "oad.parquet"))
    assert out.loc[out["Site"] == "Kalaloch", "oad_score"].iloc[0] == 1.0
    assert out.loc[out["Site"] == "Twin Harbors", "oad_score"].iloc[0] == 5.0
    assert (out["oad_overall_score"] == 3.0).all()


def test_add_oad_features_with_cloud_parquet(synthetic_oad, tmp_path):
    rows = []
    for region in ["SW Washington / Long Beach", "Overall (WA–OR–N. CA coastal)"]:
        rows.append(synthetic_oad.assign(region=region, method="ae3d", aggregation="mean"))
    pd.concat(rows, ignore_index=True).to_parquet(tmp_path / "oad.parquet")

    cloud_rows = []
    for region, val in [("SW Washington / Long Beach", 0.7),
                        ("Overall (WA–OR–N. CA coastal)", 0.5)]:
        cloud_rows.append(pd.DataFrame({
            "date": synthetic_oad["date"], "region": region,
            "valid_frac": [val] * len(synthetic_oad),
        }))
    pd.concat(cloud_rows, ignore_index=True).to_parquet(tmp_path / "cloud.parquet")

    compiled = pd.DataFrame({
        "Date": pd.to_datetime(["2013-06-10"]),
        "Site": ["Twin Harbors"],
    })
    out = add_oad_features(
        compiled, str(tmp_path / "oad.parquet"),
        cloud_parquet_path=str(tmp_path / "cloud.parquet"),
    )
    assert out.loc[0, "oad_valid_frac"] == pytest.approx(0.7)
    assert out.loc[0, "oad_overall_valid_frac"] == pytest.approx(0.5)

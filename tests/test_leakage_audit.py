"""
test_leakage_audit.py — Formalized leakage guarantee tests for DATect.

Verifies that the core data pipeline functions never expose future information
to the model training or feature building steps.

Run with: pytest tests/test_leakage_audit.py -v

NOTE: Heavy ML dependencies (xgboost, sklearn, joblib) are stubbed out so
these tests run anywhere without a full ML environment — they only need
pandas and numpy, which are always available.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub heavy dependencies before the forecasting package is imported.
# forecasting/__init__.py eagerly imports RawForecastEngine which pulls in
# xgboost, sklearn, tqdm, joblib. These stubs satisfy those imports without
# needing a full ML environment installed.
# ---------------------------------------------------------------------------

def _stub_module(name: str) -> types.ModuleType:
    """Create a stub module whose attribute access returns MagicMock objects."""
    m = types.ModuleType(name)
    m.__getattr__ = lambda attr: MagicMock()  # type: ignore[attr-defined]
    sys.modules[name] = m
    return m


_HEAVY_DEPS = [
    "tqdm", "xgboost", "joblib",
    "sklearn", "sklearn.ensemble", "sklearn.linear_model", "sklearn.preprocessing",
    "sklearn.impute", "sklearn.pipeline", "sklearn.metrics", "sklearn.compose",
    "sklearn.base", "sklearn.utils", "sklearn.utils.class_weight",
    "sklearn.utils.validation", "scipy", "scipy.stats",
]
for _dep in _HEAVY_DEPS:
    if _dep not in sys.modules:
        _stub_module(_dep)

# Now we can safely import from the forecasting package
sys.path.insert(0, str(Path(__file__).parent.parent))

from forecasting.raw_data_forecaster import (  # noqa: E402
    get_site_training_frame,
    recompute_test_row_persistence_features,
)
from forecasting.raw_data_processor import RawDataProcessor  # noqa: E402
from forecasting.raw_forecast_engine import _verify_no_data_leakage  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_site_df(n_rows: int = 20, site: str = "Newport") -> pd.DataFrame:
    """Create a minimal synthetic site DataFrame matching DATect schema."""
    dates = pd.date_range("2010-01-01", periods=n_rows, freq="7D")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "site": site,
        "date": dates,
        "da_raw": rng.uniform(0, 30, n_rows),
        "da": rng.uniform(0, 30, n_rows),
    })


def _make_multi_site_df(n_rows_per_site: int = 15) -> pd.DataFrame:
    frames = [_make_site_df(n_rows=n_rows_per_site, site=s)
              for s in ["Newport", "Copalis"]]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 1. get_site_training_frame — anchor-date enforcement
# ---------------------------------------------------------------------------

class TestGetSiteTrainingFrame:
    def test_no_rows_after_anchor(self):
        """Training frame must never include dates strictly after anchor_date."""
        df = _make_site_df(n_rows=20)
        anchor = df["date"].iloc[10]

        train = get_site_training_frame(df, site="Newport", anchor_date=anchor)
        assert train is not None, "Should produce training data"
        assert (train["date"] <= anchor).all(), (
            "Training frame contains dates after anchor — temporal leakage!"
        )

    def test_row_at_anchor_is_included(self):
        """Rows with date == anchor_date are part of training (anchor is inclusive)."""
        df = _make_site_df(n_rows=20)
        # Use row 12 as anchor so there are 13 rows (≥ default min 10) before and at anchor
        anchor = df["date"].iloc[12]

        train = get_site_training_frame(df, site="Newport", anchor_date=anchor)
        assert train is not None
        assert (train["date"] == anchor).any(), (
            "Row at anchor_date should be included in training"
        )

    def test_row_after_anchor_excluded(self):
        """A row one week after anchor must not appear in training."""
        df = _make_site_df(n_rows=20)
        anchor = df["date"].iloc[12]
        post_anchor_date = df["date"].iloc[13]

        train = get_site_training_frame(df, site="Newport", anchor_date=anchor)
        assert train is not None
        assert post_anchor_date not in train["date"].values, (
            "Row after anchor appeared in training frame — leak detected!"
        )

    def test_none_returned_for_insufficient_data(self):
        """Should return None rather than a tiny frame that causes overfitting."""
        df = _make_site_df(n_rows=5)
        anchor = df["date"].iloc[2]
        train = get_site_training_frame(
            df, site="Newport", anchor_date=anchor, min_training_samples=10
        )
        assert train is None


# ---------------------------------------------------------------------------
# 2. recompute_test_row_persistence_features — uses training data only
# ---------------------------------------------------------------------------

class TestRecomputePersistenceFeatures:
    def _make_pair(self, n_train: int = 15, spike_threshold: float = 20.0):
        df = _make_site_df(n_rows=n_train + 1)
        train_data = df.iloc[:n_train].copy()
        test_row = df.iloc[n_train:].copy()
        return train_data, test_row, spike_threshold

    def test_last_observed_da_comes_from_training_only(self):
        """last_observed_da_raw must equal the last training row's da_raw, not the test row."""
        train_data, test_row, threshold = self._make_pair()
        test_row = test_row.copy()
        test_row["da_raw"] = 999.0  # very different from training tail

        result = recompute_test_row_persistence_features(test_row, train_data, threshold)
        expected_last_da = float(train_data["da_raw"].iloc[-1])
        actual = float(result["last_observed_da_raw"].iloc[0])

        assert actual == pytest.approx(expected_last_da), (
            f"last_observed_da_raw={actual} does not match last training da_raw={expected_last_da}. "
            "Test row's da_raw is leaking into persistence features."
        )

    def test_weeks_since_spike_uses_training_only(self):
        """weeks_since_last_spike must be computed from training dates, ignoring test row."""
        train_data, test_row, _ = self._make_pair(spike_threshold=5.0)
        train_data = train_data.copy()
        train_data.iloc[-1, train_data.columns.get_loc("da_raw")] = 50.0  # spike in training
        test_row = test_row.copy()
        test_row["da_raw"] = 1.0  # not a spike

        result = recompute_test_row_persistence_features(
            test_row, train_data, spike_threshold=5.0
        )
        weeks = float(result["weeks_since_last_spike"].iloc[0])

        # Expect: (test_date - last_training_date) / 7 ≈ 1.0 week
        expected = (test_row["date"].iloc[0] - train_data["date"].iloc[-1]).days / 7.0
        assert weeks == pytest.approx(expected, abs=0.1), (
            f"weeks_since_last_spike={weeks} doesn't match expected {expected:.2f}. "
            "Spike date may be computed from test data."
        )

    def test_no_spike_gives_sentinel_value(self):
        """If no spike in training data, weeks_since_last_spike should be the sentinel 999."""
        train_data, test_row, threshold = self._make_pair()
        train_data = train_data.copy()
        train_data["da_raw"] = 0.1  # all below threshold

        result = recompute_test_row_persistence_features(test_row, train_data, threshold)
        weeks = float(result["weeks_since_last_spike"].iloc[0])
        assert weeks == 999.0, (
            f"Expected sentinel 999.0 when no spikes in training, got {weeks}"
        )


# ---------------------------------------------------------------------------
# 3. create_raw_lag_features — strictly past observations only
# ---------------------------------------------------------------------------

class TestCreateRawLagFeatures:
    def test_lag_uses_strictly_past_observations(self):
        """
        da_raw_prev_obs_1 for each row must be from a past observation,
        never the current row's value.

        This guards against the < vs <= boundary: if obs_dates <= row_date were used,
        the current measurement would predict itself (perfect information leakage).
        """
        df = _make_multi_site_df(n_rows_per_site=10)
        processor = RawDataProcessor()
        result = processor.create_raw_lag_features(
            df, group_col="site", value_col="da_raw", lags=[1, 2, 3, 4]
        )

        for site in result["site"].unique():
            site_result = (
                result[result["site"] == site]
                .sort_values("date")
                .reset_index(drop=True)
            )
            for i, row in site_result.iterrows():
                lag1 = row.get("da_raw_prev_obs_1", np.nan)
                if pd.isna(lag1):
                    continue  # first row, no history — correct
                current_da = row["da_raw"]
                assert lag1 != current_da or i == 0, (
                    f"Site {site} row {i}: lag-1 value {lag1} matches current da_raw "
                    f"{current_da}. Suggests current-row leakage in lag computation."
                )

    def test_first_row_has_no_lag(self):
        """The very first observation for a site cannot have a lag-1 (no history exists)."""
        df = _make_site_df(n_rows=8)
        processor = RawDataProcessor()
        result = processor.create_raw_lag_features(
            df, group_col="site", value_col="da_raw", lags=[1]
        )
        first_row = result.sort_values("date").iloc[0]
        assert pd.isna(first_row["da_raw_prev_obs_1"]), (
            "First row should have NaN for da_raw_prev_obs_1 (no prior observations)"
        )

    def test_lag1_equals_previous_actual_observation(self):
        """Verify lag-1 equals the immediately preceding non-NaN observation."""
        n = 6
        dates = pd.date_range("2015-01-01", periods=n, freq="7D")
        da_values = [float(v * 10) for v in range(1, n + 1)]  # 10, 20, ..., 60
        df = pd.DataFrame({
            "site": "Newport",
            "date": dates,
            "da_raw": da_values,
        })
        processor = RawDataProcessor()
        result = (
            processor.create_raw_lag_features(
                df, group_col="site", value_col="da_raw", lags=[1, 2]
            )
            .sort_values("date")
            .reset_index(drop=True)
        )

        # Row 1: lag-1 should be 10 (value from row 0)
        assert result.at[1, "da_raw_prev_obs_1"] == pytest.approx(10.0)
        # Row 2: lag-1 should be 20 (value from row 1)
        assert result.at[2, "da_raw_prev_obs_1"] == pytest.approx(20.0)
        # Row 2: lag-2 should be 10 (value from row 0)
        assert result.at[2, "da_raw_prev_obs_2"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 4. _verify_no_data_leakage — guard function correctness
# ---------------------------------------------------------------------------

class TestVerifyNoDataLeakage:
    def _make_train_df(self, max_date: str) -> pd.DataFrame:
        return pd.DataFrame({"date": [pd.Timestamp(max_date)]})

    def test_clean_case_passes(self):
        """No exception when training data is strictly within anchor."""
        train = self._make_train_df("2015-06-01")
        # Should not raise
        _verify_no_data_leakage(
            train,
            test_date="2015-07-01",
            anchor_date="2015-06-07",
        )

    def test_training_date_at_anchor_passes(self):
        """Anchor date itself is included in training — must pass."""
        train = self._make_train_df("2015-06-07")
        _verify_no_data_leakage(
            train,
            test_date="2015-07-01",
            anchor_date="2015-06-07",
        )

    def test_training_date_after_anchor_raises(self):
        """Any training row after anchor_date must raise AssertionError."""
        train = self._make_train_df("2015-06-14")  # one week past anchor
        with pytest.raises(AssertionError, match="TEMPORAL LEAK"):
            _verify_no_data_leakage(
                train,
                test_date="2015-07-01",
                anchor_date="2015-06-07",
            )

    def test_test_date_at_anchor_raises(self):
        """Test date must be strictly after anchor_date; equal is not allowed."""
        train = self._make_train_df("2015-06-01")
        with pytest.raises(AssertionError, match="TEMPORAL LEAK"):
            _verify_no_data_leakage(
                train,
                test_date="2015-06-07",  # same as anchor
                anchor_date="2015-06-07",
            )

    def test_test_date_before_anchor_raises(self):
        """Test date before anchor_date is nonsensical — must raise."""
        train = self._make_train_df("2015-06-01")
        with pytest.raises(AssertionError, match="TEMPORAL LEAK"):
            _verify_no_data_leakage(
                train,
                test_date="2015-06-01",
                anchor_date="2015-06-07",
            )

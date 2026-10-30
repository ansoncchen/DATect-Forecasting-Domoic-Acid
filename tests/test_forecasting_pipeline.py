"""
Integration tests for the core DATect forecasting pipeline and OAD feature integration.

Verifies that the RawForecastEngine can initialize successfully, build features
from raw and processed datasets, handle OAD features cleanly without leakage,
and generate regression/classification predictions correctly.
"""

from pathlib import Path
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

import config
from forecasting import ForecastEngine
from forecasting.raw_data_forecaster import load_raw_da_measurements


@pytest.mark.usefixtures("fast_models")
class TestForecastingPipeline:
    test_sample_fraction = 0.002

    @classmethod
    def setup_class(cls):
        # Verify processed parquet exists before running integration tests
        cls.parquet_path = _REPO_ROOT / "data" / "processed" / "final_output.parquet"
        assert cls.parquet_path.exists(), "Processed final_output.parquet must exist for integration tests"
        cls.engine = ForecastEngine(data_file=str(cls.parquet_path), validate_on_init=True)

    def test_load_raw_measurements(self):
        """Verify raw DA measurements load correctly and site names are canonicalized."""
        df = load_raw_da_measurements()
        assert not df.empty
        assert "date" in df.columns
        assert "site" in df.columns
        assert "da_raw" in df.columns
        assert "Twin Harbors" in df["site"].unique()

    def test_raw_measurements_one_row_per_site_day(self):
        """Same-day samples collapse to their max, so the test target is not arbitrary."""
        df = load_raw_da_measurements()
        assert not df.duplicated(["site", "date"]).any()
        # gold-beach-da.csv has two samples on 2019-01-24: 0 and 190 ug/g
        day = df[(df["site"] == "Gold Beach") & (df["date"] == "2019-01-24")]
        assert day["da_raw"].tolist() == [190.0]

    def test_regression_forecast_ensemble(self):
        """Test end-to-end regression forecasting using the ML Ensemble."""
        # Pick a valid date from Twin Harbors where history exists
        forecast_date = "2018-06-11"
        site = "Twin Harbors"
        
        result = self.engine.generate_single_forecast(
            data_path=str(self.parquet_path),
            forecast_date=forecast_date,
            site=site,
            task="regression",
            model_type="ensemble"
        )
        
        assert result is not None, f"Failed to generate forecast for {site} at {forecast_date}"
        assert "predicted_da" in result
        assert result["site"] == site
        assert result["task"] == "regression"
        assert result["model_type"] == "ensemble"
        
        # Verify ensemble weights are populated and sum to 1
        assert "ensemble_weights" in result
        assert len(result["ensemble_weights"]) == 3
        assert sum(result["ensemble_weights"]) == pytest.approx(1.0)
        
        # Verify all models made predictions
        assert "xgb_prediction" in result
        assert "rf_prediction" in result
        assert "naive_prediction" in result
        assert "ensemble_prediction" in result
        
        # Verify bootstrap quantiles are computed
        assert "bootstrap_quantiles" in result
        assert "q05" in result["bootstrap_quantiles"]
        assert "q50" in result["bootstrap_quantiles"]
        assert "q95" in result["bootstrap_quantiles"]

    def test_classification_forecast(self):
        """Test end-to-end classification forecasting producing risk categories."""
        forecast_date = "2018-06-11"
        site = "Twin Harbors"
        
        result = self.engine.generate_single_forecast(
            data_path=str(self.parquet_path),
            forecast_date=forecast_date,
            site=site,
            task="classification",
            model_type="ensemble"
        )
        
        assert result is not None
        assert "predicted_category" in result
        assert result["predicted_category"] in [0, 1, 2, 3]  # Low, Moderate, High, Extreme
        assert "class_probabilities" in result
        assert len(result["class_probabilities"]) == 4
        assert sum(result["class_probabilities"]) == pytest.approx(1.0)

    def test_spike_classifier_probability(self):
        """Verify that the dedicated spike binary classifier evaluates probability and triggers alerts."""
        if config.SPIKE_CLASSIFIER_ENABLED:
            forecast_date = "2018-06-11"
            site = "Twin Harbors"
            
            result = self.engine.generate_single_forecast(
                data_path=str(self.parquet_path),
                forecast_date=forecast_date,
                site=site,
                task="regression",
                model_type="ensemble"
            )
            
            assert "spike_probability" in result
            assert 0.0 <= result["spike_probability"] <= 1.0
            assert "spike_alert" in result
            assert isinstance(result["spike_alert"], bool)

    def test_retrospective_evaluation_smoke(self):
        """Run a lightweight retrospective evaluation to verify temporal splits and evaluation metrics."""
        # test_sample_fraction keeps this to roughly one test point per site
        eval_df = self.engine.run_retrospective_evaluation(
            task="regression",
            model_type="ensemble",
        )

        assert isinstance(eval_df, pd.DataFrame)
        assert not eval_df.empty
        required_cols = ["date", "site", "anchor_date", "actual_da", "predicted_da"]
        for col in required_cols:
            assert col in eval_df.columns

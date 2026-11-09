"""
Integration tests for the core DATect forecasting pipeline and OAD feature integration.

Verifies that the RawForecastEngine can initialize successfully, build features
from raw and processed datasets, handle OAD features cleanly without leakage,
and generate regression predictions and their risk categories correctly.
"""

import inspect
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

    def test_single_forecast_carries_no_interval(self):
        """The engine computes no interval: the dashboard shows the conformal band."""
        result = self.engine.generate_single_forecast(
            data_path=str(self.parquet_path),
            forecast_date="2018-06-11",
            site="Newport",
            task="regression",
            model_type="ensemble",
        )
        assert result is not None
        assert "bootstrap_quantiles" not in result

    def test_unknown_model_is_rejected(self):
        """A model_type no caller passes fails loudly instead of falling back to the ensemble."""
        for model_type in ("bogus", "threshold"):
            with pytest.raises(ValueError, match=f"Unknown model_type '{model_type}'"):
                self.engine.generate_single_forecast(
                    data_path=str(self.parquet_path),
                    forecast_date="2018-06-11",
                    site="Newport",
                    task="regression",
                    model_type=model_type,
                )
            with pytest.raises(ValueError, match=f"Unknown model_type '{model_type}'"):
                self.engine.run_retrospective_evaluation(task="regression", model_type=model_type)

    def test_model_selection_has_no_catch_all(self, monkeypatch):
        """A model admitted by the whitelist but not handled by the selection chain
        raises instead of being served as the linear model."""
        from forecasting import raw_forecast_engine

        monkeypatch.setattr(raw_forecast_engine, "SUPPORTED_MODELS",
                            raw_forecast_engine.SUPPORTED_MODELS + ("unhandled",))
        with pytest.raises(ValueError, match="Unknown model_type 'unhandled'"):
            self.engine.generate_single_forecast(
                data_path=str(self.parquet_path),
                forecast_date="2018-06-11",
                site="Newport",
                task="regression",
                model_type="unhandled",
            )

    def test_single_forecast_goes_through_the_point_function(self, monkeypatch):
        """The dashboard's forecast runs the per-point function the retrospective scores."""
        from forecasting.raw_forecast_engine import RawForecastEngine

        calls = []
        original = RawForecastEngine._point_forecast
        signature = inspect.signature(original)

        def spy(engine, *args, **kwargs):
            # bind, so the measurement is found however it is passed
            calls.append(signature.bind(engine, *args, **kwargs).arguments["raw_measurement"])
            return original(engine, *args, **kwargs)

        monkeypatch.setattr(RawForecastEngine, "_point_forecast", spy)
        result = self.engine.generate_single_forecast(
            data_path=str(self.parquet_path),
            forecast_date="2018-06-11",
            site="Newport",
            task="regression",
            model_type="ensemble",
        )

        assert result is not None
        assert len(calls) == 1
        assert calls[0]["site"] == "Newport"
        assert calls[0]["date"] == pd.Timestamp("2018-06-11")

    def test_classification_task_is_rejected(self):
        """The 4-category classifier and the logistic model are gone: asking for
        either fails loudly instead of falling through to another model."""
        for task, model_type, message in (("classification", "ensemble", "task 'classification'"),
                                          ("regression", "logistic", "'logistic'")):
            with pytest.raises(ValueError, match=message):
                self.engine.generate_single_forecast(
                    data_path=str(self.parquet_path),
                    forecast_date="2018-06-11",
                    site="Twin Harbors",
                    task=task,
                    model_type=model_type,
                )
            with pytest.raises(ValueError, match=message):
                self.engine.run_retrospective_evaluation(task=task, model_type=model_type)

    def test_retrospective_columns_are_pinned(self):
        """The regression retrospective's columns, in order: the always-empty logistic
        category column is gone, the Ridge competitor stays."""
        eval_df = self.engine.run_retrospective_evaluation(
            task="regression", model_type="ensemble", max_test_samples=3,
        )
        assert list(eval_df.columns) == [
            "date", "anchor_date", "processed_test_date", "site", "actual_da",
            "predicted_da", "predicted_da_rf", "naive_prediction", "predicted_da_linear",
            "training_samples", "days_ahead", "date_diff_to_processed", "feature_importance",
            "spike_probability", "spike_alert", "ensemble_prediction", "actual_category",
            "predicted_category",
        ]

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


def test_interval_knobs_are_gone():
    """The quantile and bootstrap intervals were removed with their settings."""
    for name in ("ENABLE_BOOTSTRAP_INTERVALS", "N_BOOTSTRAP_ITERATIONS",
                 "BOOTSTRAP_SUBSAMPLE_FRACTION", "CONFIDENCE_PERCENTILES",
                 "ENABLE_QUANTILE_INTERVALS"):
        assert not hasattr(config, name), name


def test_supported_models_are_pinned():
    """The engine admits exactly these models (``xgb`` is the eval scripts' alias for
    XGBoost); anything else, ``logistic`` included, is a ValueError."""
    from forecasting.raw_forecast_engine import SUPPORTED_MODELS, _require_regression

    assert SUPPORTED_MODELS == ("ensemble", "xgboost", "xgb", "rf", "naive", "linear")
    for model_type in SUPPORTED_MODELS:
        _require_regression("regression", model_type)
    with pytest.raises(ValueError, match="'logistic' was removed.*'linear'"):
        _require_regression("regression", "logistic")
    for model_type in ("threshold", "bogus", ""):
        with pytest.raises(ValueError, match="Unknown model_type"):
            _require_regression("regression", model_type)

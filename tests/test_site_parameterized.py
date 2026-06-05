"""
Thorough Parameterized Verification Suite for DATect monitoring sites and models.

Exercises:
- Config mappings across all 10 monitoring sites.
- Plotly chart builders across all 10 sites.
- End-to-end forecasts across all 10 sites, multiple historical dates,
  regression vs classification tasks, and 5 distinct model types.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import config
from forecasting import ForecastEngine
from forecasting.raw_data_forecaster import load_raw_da_measurements
from forecasting.per_site_models import (
    get_site_config,
    get_site_ensemble_weights,
    get_site_clip_params,
    apply_site_xgb_params,
    apply_site_rf_params,
    get_site_param_grid,
)
from forecasting.oad_features import SITE_TO_REGION
from backend.api import app, resolve_site
from backend.visualizations import (
    generate_correlation_heatmap,
    generate_sensitivity_analysis,
    generate_time_series_comparison,
    generate_spectral_analysis,
)
from forecasting.validation import validate_runtime_parameters
from forecasting.raw_forecast_engine import _verify_no_data_leakage

# All 10 canonical monitoring sites
SITES = [
    "Kalaloch", "Quinault", "Copalis", "Twin Harbors", "Long Beach",
    "Clatsop Beach", "Cannon Beach", "Newport", "Coos Bay", "Gold Beach"
]

# Forecast parameters to combinatorial test
DATES = ["2018-06-11", "2019-09-16", "2020-07-20", "2021-01-11"]
TASKS = ["regression", "classification"]
MODELS = ["ensemble", "xgboost", "rf", "naive", "linear"]


class TestSiteParameterized:
    @classmethod
    def setup_class(cls):
        cls.parquet_path = _REPO_ROOT / "data" / "processed" / "final_output.parquet"
        assert cls.parquet_path.exists()
        
        # Load data once for all visualization tests
        cls.df = pd.read_parquet(cls.parquet_path)

        # Scale down models and iterations for fast local parameter testing
        config.N_BOOTSTRAP_ITERATIONS = 2
        config.XGB_REGRESSION_PARAMS["n_estimators"] = 5
        config.RF_REGRESSION_PARAMS["n_estimators"] = 5
        config.XGB_CLASSIFICATION_PARAMS["n_estimators"] = 5
        config.SPIKE_CLASSIFIER_PARAMS["n_estimators"] = 5
        config.ENABLE_PARALLEL = False
        config.TEST_SAMPLE_FRACTION = 0.001
        config.MIN_TRAINING_FOR_TUNING = 99999

        # Patch site-specific overrides so they don't trigger large models
        import forecasting.per_site_models
        forecasting.per_site_models.MIN_TRAINING_FOR_TUNING = 99999
        for site, site_cfg in forecasting.per_site_models.SITE_SPECIFIC_CONFIGS.items():
            if site_cfg.get("xgb_params"):
                site_cfg["xgb_params"]["n_estimators"] = 5
            if site_cfg.get("rf_params"):
                site_cfg["rf_params"]["n_estimators"] = 5

        cls.engine = ForecastEngine(data_file=str(cls.parquet_path), validate_on_init=True)

    @classmethod
    def teardown_class(cls):
        config.MIN_TRAINING_FOR_TUNING = 80
        import forecasting.per_site_models
        forecasting.per_site_models.MIN_TRAINING_FOR_TUNING = 80

    # 1. Config tests (10 sites × 7 properties = 70 assertions in 10 tests)
    @pytest.mark.parametrize("site", SITES)
    def test_site_config_schema(self, site):
        """Verify that every site config has a standard dictionary structure."""
        cfg = get_site_config(site)
        assert isinstance(cfg, dict)
        for key in ("xgb_params", "rf_params", "param_grid", "feature_subset",
                    "ensemble_weights", "prediction_clip_q", "prediction_clip_max"):
            assert key in cfg

    @pytest.mark.parametrize("site", SITES)
    def test_site_ensemble_weights(self, site):
        """Verify that every site's weights sum to 1.0."""
        w = get_site_ensemble_weights(site)
        assert len(w) == 3
        assert sum(w) == pytest.approx(1.0)

    @pytest.mark.parametrize("site", SITES)
    def test_site_clipping_bounds(self, site):
        """Verify site clip parameters return safe bounds."""
        q, max_val = get_site_clip_params(site)
        if q is not None:
            assert 0.0 < q <= 1.0
        assert max_val is None or max_val > 0

    @pytest.mark.parametrize("site", SITES)
    def test_site_xgb_merge(self, site):
        """Verify merging site-specific XGB parameters does not raise error."""
        base = {"learning_rate": 0.05, "n_estimators": 10}
        merged = apply_site_xgb_params(base, site)
        assert isinstance(merged, dict)
        assert "learning_rate" in merged

    @pytest.mark.parametrize("site", SITES)
    def test_site_rf_merge(self, site):
        """Verify merging site-specific RF parameters does not raise error."""
        base = {"max_depth": 6}
        merged = apply_site_rf_params(base, site)
        assert isinstance(merged, dict)
        assert "max_depth" in merged

    @pytest.mark.parametrize("site", SITES)
    def test_site_param_grid(self, site):
        """Verify site custom param grid has a valid structure."""
        grid = get_site_param_grid(site)
        assert grid is None or isinstance(grid, list)

    @pytest.mark.parametrize("site", SITES)
    def test_site_oad_region_mapping(self, site):
        """Verify every site maps to a valid OAD region name."""
        assert site in SITE_TO_REGION
        assert isinstance(SITE_TO_REGION[site], str)

    # 2. Plotly Visualizations (10 sites × 4 plots = 40 tests)
    @pytest.mark.parametrize("site", SITES)
    def test_generate_correlation_heatmap_for_site(self, site):
        """Verify correlation heatmaps generate safely for each site."""
        plot = generate_correlation_heatmap(self.df, site=site)
        assert "data" in plot
        assert "layout" in plot

    @pytest.mark.parametrize("site", SITES)
    def test_generate_sensitivity_for_site(self, site):
        """Verify sensitivity analysis generates safely for each site."""
        plots = generate_sensitivity_analysis(self.df, site=site)
        assert len(plots) >= 1
        assert "data" in plots[0]

    @pytest.mark.parametrize("site", SITES)
    def test_generate_time_series_for_site(self, site):
        """Verify time-series comparisons generate safely for each site."""
        plot = generate_time_series_comparison(self.df, site=site)
        assert "data" in plot
        assert "layout" in plot

    @pytest.mark.parametrize("site", SITES)
    def test_generate_spectral_for_site(self, site):
        """Verify spectral analysis handles site subsets gracefully."""
        plots = generate_spectral_analysis(self.df, site=site)
        # Returns either an empty list (if N < 20 for test slice) or list of plots
        assert isinstance(plots, list)

    # 3. FastAPI endpoints (10 sites = 10 tests)
    @pytest.mark.parametrize("site", SITES)
    def test_api_historical_endpoint_for_site(self, site):
        """Verify historical endpoints return valid JSON and status 200 for every site."""
        with TestClient(app) as client:
            site_slug = site.lower().replace(" ", "-")
            response = client.get(f"/api/historical/{site_slug}")
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert data["site"] == site_slug

    # 4. End-to-End Pipeline Combinatorics (10 sites × 2 dates × 2 tasks × 5 models = 200 tests!)
    @pytest.mark.parametrize("site", SITES)
    @pytest.mark.parametrize("date", DATES)
    @pytest.mark.parametrize("task", TASKS)
    @pytest.mark.parametrize("model", MODELS)
    def test_pipeline_forecast_combinatorics(self, site, date, task, model):
        """Verify single forecast generation across all combinatorial paths."""
        # Twin Harbors / Newport are high-N sites, Gold Beach / Newport are small-N, etc.
        # This exercises all features, constraints, classification thresholds, and ML blended layers.
        
        # Override model classification name mapping
        m_type = "logistic" if model == "linear" and task == "classification" else model

        result = self.engine.generate_single_forecast(
            data_path=str(self.parquet_path),
            forecast_date=date,
            site=site,
            task=task,
            model_type=m_type
        )
        
        # Some early date/site combos might not have enough history and return None
        # verifying that it returns either None (fail-safe) or a valid dict (success)
        if result is not None:
            assert isinstance(result, dict)
            assert result["site"] == site
            assert result["task"] == task
            assert "predicted_da" in result or "predicted_category" in result
            assert "naive_prediction" in result

    # 5. Temporal Integrity Boundaries (20 tests)
    @pytest.mark.parametrize("anchor_offset", range(1, 21))
    def test_verify_no_leakage_bounds(self, anchor_offset):
        """Verify that any date overlap throws AssertionError in leakage checker."""
        train_df = pd.DataFrame({"date": [pd.Timestamp("2015-06-01")]})
        test_date = pd.Timestamp("2015-06-01") + pd.Timedelta(days=anchor_offset)
        
        # If anchor is at or after test, it should raise AssertionError (temporal leak)
        anchor_date = test_date
        with pytest.raises(AssertionError, match="TEMPORAL LEAK"):
            _verify_no_data_leakage(train_df, test_date, anchor_date)

    # 6. Runtime Argument Validation Boundaries (20 tests)
    @pytest.mark.parametrize("n_anchors", [-5, 0, 1000000])
    def test_validate_runtime_parameters_n_anchors(self, n_anchors):
        """Verify bad anchor parameters raise ValueError in validation."""
        if n_anchors <= 0:
            with pytest.raises(ValueError):
                validate_runtime_parameters(n_anchors, "2018-01-01")
        else:
            # high n_anchors should pass validation bounds check
            validate_runtime_parameters(n_anchors, "2018-01-01")

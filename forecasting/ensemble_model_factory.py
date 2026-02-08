"""
Ensemble Model Factory
======================

Class-based wrapper that provides the same ``get_supported_models()`` /
``get_model_description()`` interface the API layer expects, while
delegating to the raw-pipeline's standalone model builders for the actual
XGBoost / Random Forest / classifier construction.

This replaces the target project's original ``ModelFactory`` for the new
raw-data ensemble pipeline.
"""

from __future__ import annotations

from typing import Optional

import config
from .raw_model_factory import (
    build_xgb_regressor,
    build_rf_regressor,
    build_xgb_classifier,
    build_linear_regressor,
    build_logistic_classifier,
)
from .per_site_models import (
    apply_site_xgb_params,
    apply_site_rf_params,
)
from .logging_config import get_logger

logger = get_logger(__name__)


class EnsembleModelFactory:
    """
    Factory for creating configured ML models for the raw-data ensemble pipeline.

    Supported regression models:
      - ensemble: 3-model blend (XGBoost + Random Forest + Naive)
      - naive:    Last-known-DA baseline
      - linear:   Linear regression baseline

    Supported classification models:
      - ensemble:  Threshold classification from ensemble regression output
      - naive:     Threshold classification from naive regression output
      - logistic:  Logistic regression classifier
    """

    def __init__(self):
        self.random_seed = config.RANDOM_SEED

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    def get_model(
        self,
        task: str,
        model_type: str,
        site: Optional[str] = None,
        params_override: Optional[dict] = None,
    ):
        """
        Return a scikit-learn-compatible model instance.

        For ``model_type="ensemble"`` or ``"naive"`` this returns *None*
        because the ensemble blending / naive baseline are handled at the
        engine level, not as standalone estimators.

        Parameters
        ----------
        task : str
            ``"regression"`` or ``"classification"``.
        model_type : str
            One of the keys from ``get_supported_models()[task]``.
        site : str, optional
            Site name for per-site hyperparameter overrides.
        params_override : dict, optional
            Extra params merged on top of config + per-site defaults.
        """
        if task == "regression":
            return self._get_regression_model(model_type, site, params_override)
        elif task == "classification":
            return self._get_classification_model(model_type, site, params_override)
        else:
            raise ValueError(f"Unknown task: {task}. Must be 'regression' or 'classification'")

    def _get_regression_model(self, model_type, site=None, params_override=None):
        if model_type in ("ensemble", "naive"):
            # Handled at engine level
            return None

        if model_type in ("xgboost", "xgb"):
            base_params = dict(config.XGB_REGRESSION_PARAMS)
            if site and getattr(config, "USE_PER_SITE_MODELS", False):
                base_params = apply_site_xgb_params(base_params, site)
            if params_override:
                base_params.update(params_override)
            return build_xgb_regressor(base_params)

        if model_type == "rf":
            base_params = dict(getattr(config, "RF_REGRESSION_PARAMS", {}))
            if site and getattr(config, "USE_PER_SITE_MODELS", False):
                base_params = apply_site_rf_params(base_params, site)
            if params_override:
                base_params.update(params_override)
            return build_rf_regressor(base_params)

        if model_type == "linear":
            return build_linear_regressor()

        raise ValueError(
            f"Unknown regression model: {model_type}. "
            f"Supported: {self.get_supported_models()['regression']}"
        )

    def _get_classification_model(self, model_type, site=None, params_override=None):
        if model_type in ("ensemble", "naive"):
            # Threshold classification is derived from regression output
            return None

        if model_type in ("xgboost", "xgb"):
            base_params = dict(config.XGB_CLASSIFICATION_PARAMS)
            if params_override:
                base_params.update(params_override)
            return build_xgb_classifier(base_params)

        if model_type == "logistic":
            return build_logistic_classifier()

        raise ValueError(
            f"Unknown classification model: {model_type}. "
            f"Supported: {self.get_supported_models()['classification']}"
        )

    # ------------------------------------------------------------------
    # API-compatible metadata
    # ------------------------------------------------------------------

    def get_supported_models(self, task: Optional[str] = None) -> dict:
        """Return supported model types, matching the API's expected interface."""
        models = {
            "regression": ["ensemble", "naive", "linear"],
            "classification": ["ensemble", "naive", "logistic"],
        }
        if task is None:
            return models
        if task in models:
            return {task: models[task]}
        raise ValueError(f"Unknown task: {task}")

    def get_model_description(self, model_type: str) -> str:
        descriptions = {
            "ensemble": "Ensemble (XGBoost + RF + Naive)",
            "xgboost": "XGBoost",
            "xgb": "XGBoost",
            "rf": "Random Forest",
            "naive": "Naive Baseline (Most recent DA before anchor date)",
            "threshold": "Threshold Classification",
            "linear": "Ridge Regression",
            "logistic": "Logistic Regression",
        }
        return descriptions.get(model_type, f"Unknown model: {model_type}")

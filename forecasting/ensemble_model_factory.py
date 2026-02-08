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

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import config
from .raw_model_factory import (
    build_xgb_regressor,
    build_rf_regressor,
    build_xgb_classifier,
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
      - xgboost:  XGBoost regressor only
      - rf:       Random Forest regressor only
      - naive:    Last-known-DA baseline

    Supported classification models:
      - ensemble:  Threshold classification from ensemble regression output
      - xgboost:   Dedicated XGBoost classifier
      - threshold: Threshold classification from single-model regression output
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

        raise ValueError(
            f"Unknown regression model: {model_type}. "
            f"Supported: {self.get_supported_models()['regression']}"
        )

    def _get_classification_model(self, model_type, site=None, params_override=None):
        if model_type in ("ensemble", "threshold"):
            # Threshold classification is derived from regression output
            return None

        if model_type in ("xgboost", "xgb"):
            base_params = dict(config.XGB_CLASSIFICATION_PARAMS)
            if params_override:
                base_params.update(params_override)
            return build_xgb_classifier(base_params)

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
            "regression": ["ensemble", "xgboost", "rf", "naive"],
            "classification": ["ensemble", "xgboost", "threshold"],
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
            "naive": "Naive Baseline (Last Known DA)",
            "threshold": "Threshold Classification",
            "linear": "Linear Regression",
            "logistic": "Logistic Regression",
        }
        return descriptions.get(model_type, f"Unknown model: {model_type}")

    # ------------------------------------------------------------------
    # Sample weighting helpers (ported from original ModelFactory)
    # ------------------------------------------------------------------

    def compute_sample_weights_for_classification(self, y_train) -> np.ndarray:
        """
        Compute balanced sample weights for classification training.

        Uses sklearn's ``compute_class_weight('balanced')`` to address
        class imbalance in DA risk categories.
        """
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
        class_weight_dict = dict(zip(unique_classes, class_weights))
        sample_weights = np.array([class_weight_dict[y] for y in y_train])
        return sample_weights

    def compute_spike_focused_weights(self, y_actual) -> np.ndarray:
        """
        Compute sample weights that heavily penalise missed spikes.

        Spike events (DA > ``config.SPIKE_THRESHOLD``) receive weight
        ``SPIKE_FALSE_NEGATIVE_WEIGHT``; non-spike events receive
        ``SPIKE_TRUE_NEGATIVE_WEIGHT``.
        """
        actual_spikes = (y_actual > config.SPIKE_THRESHOLD).astype(int)
        sample_weights = np.ones(len(actual_spikes))
        spike_mask = actual_spikes == 1
        sample_weights[spike_mask] = getattr(config, "SPIKE_FALSE_NEGATIVE_WEIGHT", 500.0)
        non_spike_mask = actual_spikes == 0
        sample_weights[non_spike_mask] = getattr(config, "SPIKE_TRUE_NEGATIVE_WEIGHT", 0.1)
        return sample_weights

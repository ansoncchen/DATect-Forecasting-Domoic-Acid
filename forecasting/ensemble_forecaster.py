"""
Ensemble Forecaster
===================

Combines naive persistence with ML models (XGBoost + optional TFT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .model_factory import ModelFactory
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleWeights:
    naive: float = 0.4
    xgboost: float = 0.4
    tft: float = 0.2

    def normalized(self) -> "EnsembleWeights":
        total = self.naive + self.xgboost + self.tft
        if total <= 0:
            return EnsembleWeights(naive=1.0, xgboost=0.0, tft=0.0)
        return EnsembleWeights(
            naive=self.naive / total,
            xgboost=self.xgboost / total,
            tft=self.tft / total,
        )


class EnsembleForecaster:
    def __init__(
        self,
        weights: Optional[EnsembleWeights] = None,
        model_factory: Optional[ModelFactory] = None,
    ):
        self.weights = (weights or EnsembleWeights()).normalized()
        self.model_factory = model_factory or ModelFactory()
        self.xgb_model = None
        self.tft_model = None

    def fit_xgboost(self, X_train, y_train, params_override=None):
        self.xgb_model = self.model_factory.get_model(
            "regression", "xgboost", params_override=params_override
        )
        self.xgb_model.fit(X_train, y_train)
        return self

    def set_tft_model(self, tft_model):
        self.tft_model = tft_model
        return self

    def _predict_xgb(self, X):
        if self.xgb_model is None:
            return None
        pred = self.xgb_model.predict(X)
        return np.maximum(0.0, pred.astype(float))

    def _predict_tft(self, X):
        if self.tft_model is None:
            return None
        pred = self.tft_model.predict(X)
        return np.maximum(0.0, np.asarray(pred, dtype=float))

    def predict(self, X, last_known_da: Sequence[float]):
        """
        Predict using weighted ensemble.

        Args:
            X: Feature matrix for ML models.
            last_known_da: Naive baseline values (last observed DA).
        """
        last_known_da = np.asarray(last_known_da, dtype=float)
        w = self.weights

        xgb_pred = self._predict_xgb(X)
        tft_pred = self._predict_tft(X)

        # Adjust weights if optional model is missing
        w_naive = w.naive
        w_xgb = w.xgboost if xgb_pred is not None else 0.0
        w_tft = w.tft if tft_pred is not None else 0.0
        total = w_naive + w_xgb + w_tft
        if total <= 0:
            return np.maximum(0.0, last_known_da)

        w_naive /= total
        w_xgb /= total
        w_tft /= total

        preds = w_naive * last_known_da
        if xgb_pred is not None:
            preds = preds + w_xgb * xgb_pred
        if tft_pred is not None:
            preds = preds + w_tft * tft_pred

        return np.maximum(0.0, preds)

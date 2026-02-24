"""
Classification Adapter
======================

Handles the dual classification approach for DA risk categories:

1. **Threshold classification** — deterministic mapping from continuous
   DA predictions to risk categories using ``config.DA_CATEGORY_BINS``.
   This is the default and most straightforward approach.

2. **Dedicated ML classifier** — trains a separate XGBoost classifier on
   the 4-category labels with balanced class weighting, and returns per-
   class probabilities.  Used for the probability bar chart in the UI.

DA Risk Categories:
  0 = Low      (0 – 5 µg/g)
  1 = Moderate  (5 – 20 µg/g)
  2 = High     (20 – 40 µg/g)
  3 = Extreme  (40+ µg/g)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

import config
from .raw_model_factory import build_xgb_classifier
from .logging_config import get_logger

logger = get_logger(__name__)


class ClassificationAdapter:
    """
    Adapter that provides both threshold-based and ML-based DA classification.
    """

    def __init__(self):
        self.bins = config.DA_CATEGORY_BINS
        self.labels = config.DA_CATEGORY_LABELS
        self.spike_threshold = config.SPIKE_THRESHOLD

    # ------------------------------------------------------------------
    # 1. Threshold classification (from regression output)
    # ------------------------------------------------------------------

    def threshold_classify(self, da_value: float) -> int:
        """
        Map a continuous DA prediction to a risk category.

        Uses ``config.DA_CATEGORY_BINS`` for the cut points:
          0 – 5   → 0 (Low)
          5 – 20  → 1 (Moderate)
          20 – 40 → 2 (High)
          40+     → 3 (Extreme)
        """
        result = pd.cut(
            [da_value],
            bins=self.bins,
            labels=self.labels,
        )
        return int(result[0])

    def threshold_classify_series(self, da_values: pd.Series) -> pd.Series:
        """Vectorised threshold classification for a column of DA values."""
        return pd.cut(da_values, bins=self.bins, labels=self.labels).astype(int)

    # ------------------------------------------------------------------
    # 2. Dedicated XGBoost classifier
    # ------------------------------------------------------------------

    def train_dedicated_classifier(
        self,
        X_train: pd.DataFrame,
        y_da_raw: pd.Series,
        site: Optional[str] = None,
        params_override: Optional[dict] = None,
    ) -> tuple:
        """
        Train a dedicated XGBoost classifier on DA risk categories.

        Parameters
        ----------
        X_train : DataFrame
            Preprocessed training features (already transformed).
        y_da_raw : Series
            Raw DA values for the training set (will be binned into categories).
        site : str, optional
            Site name (reserved for future per-site classifier tuning).
        params_override : dict, optional
            Extra parameters for the classifier.

        Returns
        -------
        (model, cat_mapping, reverse_mapping) or None if training fails.
        """
        # Create category labels from raw DA
        y_categories = pd.cut(
            y_da_raw,
            bins=self.bins,
            labels=self.labels,
        ).astype(int)

        unique_classes = np.unique(y_categories)
        if len(unique_classes) < 2:
            logger.warning(
                "Only one DA category present in training data for %s — "
                "skipping dedicated classifier",
                site or "unknown",
            )
            return None

        # Create label mappings (needed when not all 4 categories are present)
        sorted_cats = sorted(unique_classes)
        cat_mapping = {cat: i for i, cat in enumerate(sorted_cats)}
        reverse_mapping = {i: cat for cat, i in cat_mapping.items()}

        y_encoded = y_categories.map(cat_mapping)

        # Balanced class weighting
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_encoded), y=y_encoded,
        )
        class_weight_dict = dict(zip(np.unique(y_encoded), class_weights))
        sample_weights = np.array([class_weight_dict[y] for y in y_encoded])

        # Build and train classifier
        model = build_xgb_classifier(params_override)
        try:
            model.fit(X_train, y_encoded, sample_weight=sample_weights)
        except Exception as exc:
            logger.warning("Dedicated classifier training failed: %s", exc)
            return None

        return model, cat_mapping, reverse_mapping

    def predict_with_probabilities(
        self,
        model,
        X_test: pd.DataFrame,
        reverse_mapping: dict,
    ) -> tuple[int, list[float]]:
        """
        Predict category and return per-class probabilities.

        Parameters
        ----------
        model
            Trained XGBoost classifier.
        X_test : DataFrame
            Preprocessed test features (single row or batch).
        reverse_mapping : dict
            ``{encoded_label: original_category}`` from training.

        Returns
        -------
        (predicted_category, probabilities)
            ``probabilities`` is a 4-element list ``[p_low, p_mod, p_high, p_extreme]``.
        """
        pred_encoded = int(model.predict(X_test)[0])
        predicted_category = reverse_mapping[pred_encoded]

        # Build full 4-class probability array
        prob_array = [0.0, 0.0, 0.0, 0.0]
        if hasattr(model, "predict_proba"):
            raw_probs = model.predict_proba(X_test)[0]
            for i, prob in enumerate(raw_probs):
                original_cat = reverse_mapping.get(i)
                if original_cat is not None and 0 <= original_cat <= 3:
                    prob_array[original_cat] = float(prob)
        else:
            # If no probabilities available, assign 1.0 to predicted class
            prob_array[predicted_category] = 1.0

        return int(predicted_category), prob_array

    # ------------------------------------------------------------------
    # Convenience: classify + probabilities in one call
    # ------------------------------------------------------------------

    def classify_prediction(
        self,
        da_prediction: float,
        classifier_result: Optional[tuple] = None,
        X_test: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Return a classification result dict for the API response.

        Uses threshold classification as the primary method.
        If a trained classifier is provided, also returns ML-based
        category and probabilities.

        Returns
        -------
        dict with keys:
          - predicted_category: int (threshold-based)
          - predicted_category_ml: int or None (classifier-based)
          - class_probabilities: list[float] length 4
        """
        threshold_cat = self.threshold_classify(da_prediction)

        result = {
            "predicted_category": threshold_cat,
            "predicted_category_ml": None,
            "class_probabilities": [0.0, 0.0, 0.0, 0.0],
        }

        # If we have a trained classifier, use it for probabilities
        if classifier_result is not None and X_test is not None:
            model, _cat_mapping, reverse_mapping = classifier_result
            try:
                ml_cat, probs = self.predict_with_probabilities(
                    model, X_test, reverse_mapping
                )
                result["predicted_category_ml"] = ml_cat
                result["class_probabilities"] = probs
            except Exception as exc:
                logger.warning("ML classification prediction failed: %s", exc)
                # Fall back to threshold-only
                result["class_probabilities"][threshold_cat] = 1.0
        else:
            # No classifier — assign 100% probability to threshold category
            result["class_probabilities"][threshold_cat] = 1.0

        return result

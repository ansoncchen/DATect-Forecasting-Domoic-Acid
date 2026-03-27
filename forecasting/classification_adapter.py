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

from dataclasses import dataclass
from typing import Any, Optional

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

    # ------------------------------------------------------------------
    # 3. Binary spike classifier (transition-focused)
    # ------------------------------------------------------------------

    def train_spike_binary_classifier(
        self,
        X_train: pd.DataFrame,
        y_da_raw: pd.Series,
        spike_threshold: float = 20.0,
        site: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Train a binary XGBoost classifier for spike detection.

        Uses a **safe-baseline** approach: trains only on rows where the
        previous observation (``da_raw_prev_obs_1``) was below
        *spike_threshold*.  This forces the model to learn from
        environmental and rolling features rather than persistence.

        Leaky features (``last_observed_da_raw``, ``weeks_since_last_spike``)
        are dropped before fitting so the classifier cannot cheat.

        Returns
        -------
        dict with keys ``model``, ``columns`` (features used), or None.
        """
        # Binary target from raw DA values
        y_spike = (y_da_raw >= spike_threshold).astype(int)

        # Identify which columns to drop (leaky persistence features)
        # distance_to_threshold = SPIKE_THRESHOLD - last_observed_da_raw, also leaky
        leaky_cols = {"last_observed_da_raw", "weeks_since_last_spike", "distance_to_threshold"}
        prev_obs_col = "da_raw_prev_obs_1"

        # Safe-baseline: keep only rows where previous obs < threshold
        if prev_obs_col in X_train.columns:
            safe_mask = X_train[prev_obs_col].fillna(0) < spike_threshold
        else:
            # If prev_obs_col was already dropped, fall back to full training set
            safe_mask = pd.Series(True, index=X_train.index)

        X_safe = X_train.loc[safe_mask].copy()
        y_safe = y_spike.loc[safe_mask].copy()

        # Drop leaky columns from features
        cols_to_drop = [c for c in leaky_cols if c in X_safe.columns]
        if cols_to_drop:
            X_safe = X_safe.drop(columns=cols_to_drop)

        # Need at least 2 classes and minimum samples
        if len(y_safe) < 10 or y_safe.nunique() < 2:
            logger.debug(
                "Spike classifier: insufficient safe-baseline data for %s "
                "(n=%d, classes=%d)",
                site or "unknown",
                len(y_safe),
                y_safe.nunique(),
            )
            return None

        # Balanced class weighting
        classes = np.unique(y_safe)
        class_weights = compute_class_weight("balanced", classes=classes, y=y_safe)
        class_weight_dict = dict(zip(classes, class_weights))
        sample_weights = np.array([class_weight_dict[y] for y in y_safe])

        # Use spike-specific params (shallower trees, tuned for small
        # safe-baseline datasets in per-test-point training)
        from xgboost import XGBClassifier
        spike_params = getattr(config, "SPIKE_CLASSIFIER_PARAMS", {})
        model = XGBClassifier(
            **spike_params,
            random_state=config.RANDOM_SEED,
            verbosity=0,
            use_label_encoder=False,
        )
        try:
            model.fit(X_safe, y_safe, sample_weight=sample_weights)
        except Exception as exc:
            logger.warning("Spike binary classifier training failed: %s", exc)
            return None

        return {"model": model, "columns": list(X_safe.columns)}

    def predict_spike_probability(
        self,
        spike_result: dict,
        X_test: pd.DataFrame,
    ) -> float:
        """
        Predict spike probability for a test row using the trained spike
        classifier.  Aligns test columns to match training columns.

        Returns probability of class 1 (spike).
        """
        model = spike_result["model"]
        train_cols = spike_result["columns"]
        X_aligned = X_test.reindex(columns=train_cols, fill_value=0)
        proba = model.predict_proba(X_aligned)
        # Class 1 = spike; handle case where model only saw one class
        if proba.shape[1] == 1:
            return 0.0
        return float(proba[0, 1])

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

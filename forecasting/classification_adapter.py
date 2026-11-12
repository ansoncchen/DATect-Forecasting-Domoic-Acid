"""
Classification Adapter
======================

1. **Threshold classification** — deterministic mapping from continuous
   DA values (measured or predicted) to risk categories using
   ``config.DA_CATEGORY_BINS``.

2. **Binary spike classifier** — the XGBoost classifier behind the spike
   alert, trained on pooled all-site data (``train_spike_binary_classifier`` /
   ``predict_spike_probability``).

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
from .logging_config import get_logger

logger = get_logger(__name__)


class ClassificationAdapter:
    """
    Threshold risk categories and the binary spike classifier.
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

        Uses ``config.DA_CATEGORY_BINS`` as left-closed intervals:
          [0, 5)   → 0 (Low)
          [5, 20)  → 1 (Moderate)
          [20, 40) → 2 (High)
          [40, ∞)  → 3 (Extreme)
        """
        result = pd.cut(
            [da_value],
            bins=self.bins,
            labels=self.labels, right=False,
        )
        return int(result[0])

    def threshold_classify_series(self, da_values: pd.Series) -> pd.Series:
        """Vectorised threshold classification for a column of DA values."""
        return pd.cut(da_values, bins=self.bins, labels=self.labels, right=False).astype(int)

    # ------------------------------------------------------------------
    # 2. Binary spike classifier (transition-focused)
    # ------------------------------------------------------------------

    def train_spike_binary_classifier(
        self,
        X_train: pd.DataFrame,
        y_da_raw: pd.Series,
        spike_threshold: Optional[float] = None,  # default → config.SPIKE_THRESHOLD
        site: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Train a binary XGBoost classifier for spike detection.

        Trains on every row it is given: all real measurements, including
        weeks that were already above *spike_threshold*. ``X_train`` is the
        pooled matrix after MinMax scaling. An earlier "safe-baseline" filter
        compared the scaled ``da_raw_prev_obs_1`` (always in [0, 1]) with the
        threshold in ug/g, so it never removed a row. Applied to unscaled
        values it lowered validation onset AUC by 0.023, so it was removed
        rather than fixed.

        ``last_observed_da_raw`` and ``weeks_since_last_spike`` are dropped
        before fitting; the lag and rolling DA features stay.

        Returns
        -------
        dict with keys ``model``, ``columns`` (features used), or None.
        """
        # Resolve threshold default from config (single source of truth)
        if spike_threshold is None:
            spike_threshold = float(config.SPIKE_THRESHOLD)

        # Binary target from raw DA values
        y_safe = (y_da_raw >= spike_threshold).astype(int)

        # distance_to_threshold = SPIKE_THRESHOLD - last_observed_da_raw
        leaky_cols = {"last_observed_da_raw", "weeks_since_last_spike", "distance_to_threshold"}
        cols_to_drop = [c for c in leaky_cols if c in X_train.columns]
        X_safe = X_train.drop(columns=cols_to_drop)

        # Need at least 2 classes and minimum samples
        if len(y_safe) < 10 or y_safe.nunique() < 2:
            logger.debug(
                "Spike classifier: insufficient training data for %s "
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

        # Use spike-specific params (shallower trees, refit per test point)
        from xgboost import XGBClassifier
        spike_params = dict(config.SPIKE_CLASSIFIER_PARAMS)
        # One thread: the probability must not depend on the machine's core count
        # (retrospective workers were already capped at one thread by joblib).
        spike_params["n_jobs"] = 1
        model = XGBClassifier(
            **spike_params,
            random_state=config.MODEL_SEED,
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

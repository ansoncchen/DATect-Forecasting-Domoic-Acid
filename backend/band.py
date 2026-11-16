"""
The 80% error band served with every forecast.

Calibrated at request time from a committed error history: the all-years
chronological run, whose forecasts are the engine's own, so a forecast's band
uses only errors of forecasts measured by its anchor (docs/ERROR_BAND.md). The
variant (error window, Mondrian) is the one the declared protocol chose on
validation, read from its result file. Regenerate the history whenever the
engine's predictions change, as for the golden baseline.

Errors are selected exactly as ``forecasting.conformal.past_residual_bands``
selects them, so the served band equals the evaluated one
(``tests/test_band_service.py``). Mondrian selection lives here rather than in
``forecasting/`` because that package is part of the precomputed store's
config hash, and the band is computed at request time, not stored.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

import config
from forecasting.conformal import (
    CATEGORY_EDGES,
    ErrorHistory,
    band_from_residuals,
    category_probabilities,
    exceedance_probability,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERROR_HISTORY_PATH = "results/chronological/chronological_regression_ensemble_20030101_20240101.parquet"
BAND_EVAL_PATH = "results/conformal_band/band_eval.json"
MIN_CALIBRATION = 30


@lru_cache(maxsize=1)
def chosen_variant() -> dict:
    """Coverage, error window and Mondrian flag chosen on validation (band_eval.json)."""
    with open(os.path.join(REPO_ROOT, BAND_EVAL_PATH)) as f:
        evaluation = json.load(f)
    chosen = evaluation["selection"]["chosen"]
    return {"coverage": float(evaluation["coverage_target"]), "window": chosen["window"],
            "mondrian": bool(chosen.get("mondrian", False))}


def _predicted_category(predicted) -> np.ndarray:
    # Left-closed bins, as in forecasting.conformal: a forecast of exactly 5 is in [5, 20)
    return np.searchsorted(CATEGORY_EDGES, np.clip(predicted, 0.0, None), side="right")


@lru_cache(maxsize=1)
def _histories() -> tuple[ErrorHistory, dict[int, ErrorHistory]] | None:
    """Every past error, and the same errors split by the forecast's predicted category."""
    path = os.path.join(REPO_ROOT, ERROR_HISTORY_PATH)
    if not os.path.exists(path):
        return None
    frame = pd.read_parquet(path, columns=["date", "actual_da", "predicted_da"])
    category = _predicted_category(frame["predicted_da"].to_numpy(dtype=float))
    by_category = {c: ErrorHistory(frame[category == c]) for c in range(len(CATEGORY_EDGES) + 1)}
    return ErrorHistory(frame), by_category


def _calibration_errors(predicted_da: float, anchor_date, variant: dict) -> np.ndarray:
    pooled, by_category = _histories()
    window = variant["window"]
    if variant["mondrian"]:
        # Errors of past forecasts in this forecast's category, while there are enough
        same = by_category[int(_predicted_category(predicted_da))].known_by(anchor_date)
        if len(same) >= MIN_CALIBRATION:
            return same if window is None else same[-window:]
    return pooled.known_by(anchor_date, window)


def forecast_band(predicted_da: float, anchor_date) -> dict | None:
    """Band, P(DA >= spike threshold) and category chances for one forecast, or
    None when there is no history or fewer than 30 errors before the anchor."""
    if _histories() is None or predicted_da is None or not np.isfinite(predicted_da):
        return None
    variant = chosen_variant()
    errors = _calibration_errors(predicted_da, anchor_date, variant)
    if len(errors) < MIN_CALIBRATION:
        return None
    lo, hi = band_from_residuals(predicted_da, errors, variant["coverage"])
    edges = tuple(config.DA_CATEGORY_BINS[1:-1])
    return {"lo": lo, "hi": hi, "level": variant["coverage"], "n_calibration": len(errors),
            "p_exceed": exceedance_probability(predicted_da, errors, config.SPIKE_THRESHOLD),
            "category_probabilities": category_probabilities(predicted_da, errors, edges)}

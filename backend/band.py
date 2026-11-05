"""
The 80% error band served with every forecast.

Calibrated at request time from a committed error history: the all-years
chronological run, whose forecasts are the engine's own, so a forecast's band
uses only errors of forecasts measured by its anchor (docs/ERROR_BAND.md). The
variant (error window, Mondrian) is the one the declared protocol chose on
validation, read from its result file. Regenerate the history whenever the
engine's predictions change, as for the golden baseline.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import pandas as pd

import config
from forecasting.conformal import ErrorHistory, category_probabilities

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERROR_HISTORY_PATH = "results/chronological/chronological_regression_ensemble_20030101_20240101.parquet"
BAND_EVAL_PATH = "results/conformal_band/band_eval.json"
MIN_CALIBRATION = 30


@lru_cache(maxsize=1)
def chosen_variant() -> dict:
    """Coverage and error window chosen on validation (band_eval.json)."""
    with open(os.path.join(REPO_ROOT, BAND_EVAL_PATH)) as f:
        evaluation = json.load(f)
    chosen = evaluation["selection"]["chosen"]
    if chosen.get("mondrian"):
        raise ValueError("band_eval.json chose Mondrian calibration, which serving does not implement")
    return {"coverage": float(evaluation["coverage_target"]), "window": chosen["window"]}


@lru_cache(maxsize=1)
def _history() -> ErrorHistory | None:
    path = os.path.join(REPO_ROOT, ERROR_HISTORY_PATH)
    if not os.path.exists(path):
        return None
    return ErrorHistory(pd.read_parquet(path, columns=["date", "actual_da", "predicted_da"]))


def forecast_band(predicted_da: float, anchor_date) -> dict | None:
    """Band, P(DA >= spike threshold) and category chances for one forecast, or
    None when there is no history or fewer than 30 errors before the anchor."""
    history = _history()
    if history is None:
        return None
    variant = chosen_variant()
    result = history.band(predicted_da, anchor_date, coverage=variant["coverage"],
                          window=variant["window"], threshold=config.SPIKE_THRESHOLD,
                          min_calibration=MIN_CALIBRATION)
    if result is None:
        return None
    edges = tuple(config.DA_CATEGORY_BINS[1:-1])
    result["category_probabilities"] = category_probabilities(
        predicted_da, history.known_by(anchor_date, variant["window"]), edges)
    return result

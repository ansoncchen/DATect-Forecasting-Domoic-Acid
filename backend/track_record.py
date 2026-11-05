"""
Track record of the deployed model, summarised server-side so the dashboard
computes nothing in the browser (dashboard design, sections 4 and 6).

Rows are the deterministic chronological runs committed under
``results/chronological`` (every real measurement in the window), joined with the
conformal band of each forecast (``results/conformal_band/band_predictions.parquet``).
R², MAE and log R² come verbatim from the runs' JSON, with their bootstrap CIs,
so they equal the reported numbers. Everything else is counted from the rows with
the shared definitions in ``forecasting.metrics``.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

import config
from backend.band import REPO_ROOT
from forecasting.metrics import alert_parts, exceeds, onset_scores
from forecasting.tuned_config import get_eval_windows

Window = Literal["holdout", "validation"]
BAND_PREDICTIONS_PATH = "results/conformal_band/band_predictions.parquet"
ROW_COLUMNS = ["date", "anchor_date", "site", "actual_da", "predicted_da", "naive_prediction",
               "predicted_da_linear", "spike_probability", "spike_alert", "band_lo", "band_hi", "p_exceed"]
HISTORY_COLUMNS = ["date", "anchor_date", "site", "actual_da", "predicted_da", "naive_prediction",
                   "spike_probability", "spike_alert", "band_lo", "band_hi", "p_exceed"]
LEVELS = ("Quiet", "Watch", "Warning")
MODELS = {"ensemble": "predicted_da", "naive": "naive_prediction", "linear": "predicted_da_linear"}
MIN_ROWS_FOR_FIT = 3


def window_bounds(window: Window) -> tuple[str, str]:
    w = get_eval_windows()
    return w[f"{window}_start"], w[f"{window}_end"]


def _run_path(window: Window, suffix: str) -> str:
    start, end = window_bounds(window)
    name = f"chronological_regression_ensemble_{start.replace('-', '')}_{end.replace('-', '')}.{suffix}"
    return os.path.join(REPO_ROOT, "results", "chronological", name)


@lru_cache(maxsize=1)
def _band_predictions() -> pd.DataFrame:
    frame = pd.read_parquet(os.path.join(REPO_ROOT, BAND_PREDICTIONS_PATH))
    frame["date"] = pd.to_datetime(frame["date"])
    frame["anchor_date"] = pd.to_datetime(frame["anchor_date"])
    return frame


@lru_cache(maxsize=2)
def _window_frame(window: Window) -> pd.DataFrame:
    rows = pd.read_parquet(_run_path(window, "parquet"))
    rows["date"] = pd.to_datetime(rows["date"])
    rows["anchor_date"] = pd.to_datetime(rows["anchor_date"])
    band = _band_predictions()[["date", "site", "band_lo", "band_hi", "p_exceed"]]
    rows = rows.merge(band, on=["date", "site"], how="left", validate="one_to_one")
    return rows[ROW_COLUMNS].sort_values(["date", "site"], kind="stable").reset_index(drop=True)


@lru_cache(maxsize=2)
def _window_json(window: Window) -> dict:
    with open(_run_path(window, "json")) as f:
        return json.load(f)


def window_rows(window: Window, site: str | None = None) -> pd.DataFrame:
    rows = _window_frame(window)
    return (rows if site is None else rows[rows["site"] == site]).reset_index(drop=True)


def _alert_levels(rows: pd.DataFrame) -> list:
    """The alert's parts for each row (dict per row), from the engine's own outputs."""
    return [alert_parts(p, f, last, prob_cutoff=config.SPIKE_ALERT_PROB_THRESHOLD,
                        regression_cutoff=config.SPIKE_REGRESSION_ALERT_THRESHOLD,
                        warning_cutoff=config.ALERT_WARNING_PROB_CUTOFF, threshold=config.SPIKE_THRESHOLD)
            for p, f, last in zip(rows["spike_probability"], rows["predicted_da"], rows["naive_prediction"])]


def _share(mask: np.ndarray) -> float | None:
    return float(mask.mean()) if mask.size else None


def summarize(rows: pd.DataFrame) -> dict:
    """Counts and coverage over ``rows``. Fit metrics come only from ``window_summary``."""
    thr = config.SPIKE_THRESHOLD
    y = rows["actual_da"].to_numpy(dtype=float)
    last = rows["naive_prediction"].to_numpy(dtype=float)
    alert = rows["spike_alert"].astype(bool).to_numpy()
    below, jump = ~exceeds(last, thr), exceeds(y, thr)
    onset, quiet = below & jump, below & ~jump
    scores = onset_scores(last, y, alert, threshold=thr)
    false = alert & quiet

    lo, hi = rows["band_lo"].to_numpy(dtype=float), rows["band_hi"].to_numpy(dtype=float)
    has_band = ~np.isnan(lo)
    inside = (y >= lo) & (y <= hi)

    levels = {name: {"weeks": 0, "followed_by_jump": 0} for name in LEVELS}
    for parts, jumped in zip(_alert_levels(rows), jump):
        if parts["level"] is not None:
            levels[parts["level"]]["weeks"] += 1
            levels[parts["level"]]["followed_by_jump"] += int(jumped)

    return {
        "n": int(len(rows)),
        "band": {
            "coverage": _share(inside[has_band]),
            "inside": int(inside[has_band].sum()),
            "n_banded": int(has_band.sum()),
            "coverage_at_onsets": _share(inside[has_band & onset]),
            "onsets_inside": int(inside[has_band & onset].sum()),
            "median_width": float(np.median((hi - lo)[has_band])) if has_band.any() else None,
        },
        "onsets": {"n": scores["n_onsets"], "caught": scores.get("onsets_caught", 0),
                   "quiet": scores["n_quiet"], "quiet_alarmed": int(false.sum())},
        "alarms": {
            "n": int((alert & below).sum()),
            "followed_by_jump": int((alert & onset).sum()),
            "false_below_5": int((false & (y < 5)).sum()),
            "false_5_to_10": int((false & (y >= 5) & (y < 10)).sum()),
            "false_near_miss": int((false & (y >= 10)).sum()),
        },
        "levels": levels,
    }


def window_summary(window: Window, site: str | None = None) -> dict:
    """``summarize`` plus R², MAE and log R² ([v, lo, hi], from the run's JSON) and
    the ensemble / last-value / linear comparison on the same rows."""
    rows = window_rows(window, site)
    out = summarize(rows)
    run = _window_json(window)
    fit = run["overall"] if site is None else run["per_site"].get(site)
    y = rows["actual_da"].to_numpy(dtype=float)
    # R2 means nothing without variance to explain: sklearn returns 0 or 1 on a constant target
    fit_defined = fit is not None and len(y) >= MIN_ROWS_FOR_FIT and np.ptp(y) > 0
    out["r2"] = fit["r2"] if fit_defined else None
    out["mae"] = fit["mae"] if fit else None
    out["log_r2"] = fit.get("r2_log1p") if fit_defined else None
    out["models"] = {
        name: {"r2": float(r2_score(y, rows[col])), "mae": float(mean_absolute_error(y, rows[col]))}
        for name, col in MODELS.items()
    } if fit_defined else None
    return out


def records(rows: pd.DataFrame) -> list:
    """Rows as JSON-ready dicts, dates as YYYY-MM-DD."""
    out = rows.copy()
    for col in ("date", "anchor_date"):
        out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


def site_history(site: str, since: str = "2019-01-01", until: str | None = None) -> list:
    """Each real measurement at ``site`` with the forecast made for it, its band,
    the classifier probability, the alert and its level."""
    bp = _band_predictions()
    keep = (bp["site"] == site) & (bp["date"] >= pd.Timestamp(since))
    if until is not None:
        keep &= bp["date"] <= pd.Timestamp(until)
    rows = bp.loc[keep, HISTORY_COLUMNS].sort_values("date", kind="stable").reset_index(drop=True)
    parts = _alert_levels(rows)
    rows["alert"] = [p["alert"] for p in parts]
    rows["level"] = [p["level"] for p in parts]
    return records(rows)

"""
Conformal error band for the regression forecast.

The band widens a forecast by the model's own past errors. Errors are taken on
the log1p scale, where a miss of 5 ug/g matters more at 3 than at 80, so the
band is narrow when DA is low and widens multiplicatively when it is high.

Leak-free by construction: a forecast is calibrated only by earlier forecasts
whose measurement is dated on or before its anchor (``test_date - 7 days``),
the same information cut-off the model itself uses. Errors are pooled across
sites, so one rule serves every site.

Options:
- ``window``: keep only the most recent ``window`` errors (None keeps all).
- ``mondrian``: calibrate each forecast only by past forecasts in the same
  predicted category ([0,5) [5,20) [20,inf)), falling back to every past
  forecast while a category has fewer than ``min_calibration`` errors.

The same errors give the chance that the measurement reaches a threshold:
the share of past errors that would carry this forecast to it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

CATEGORY_EDGES = (5.0, 20.0)


def log_residuals(actual, predicted) -> np.ndarray:
    """Signed errors on the log1p scale (actual minus forecast)."""
    return np.log1p(np.asarray(actual, dtype=float)) - np.log1p(
        np.clip(np.asarray(predicted, dtype=float), 0.0, None))


def band_from_residuals(pred: float, residuals: np.ndarray, coverage: float) -> tuple[float, float]:
    """Split-conformal band around one forecast, in ug/g.

    Uses the finite-sample ranks floor((n+1)a/2) and ceil((n+1)(1-a/2)) with
    a = 1 - coverage; a rank outside 1..n leaves that side unbounded.
    """
    r = np.sort(residuals)
    n = len(r)
    alpha = 1.0 - coverage
    # 1 - 0.8 is 0.19999999999999996; the tolerance keeps exact ranks exact
    k_lo = math.floor((n + 1) * alpha / 2 + 1e-9)
    k_hi = math.ceil((n + 1) * (1 - alpha / 2) - 1e-9)
    base = np.log1p(max(pred, 0.0))
    lo = 0.0 if k_lo < 1 else max(0.0, float(np.expm1(base + r[k_lo - 1])))
    hi = math.inf if k_hi > n else float(np.expm1(base + r[k_hi - 1]))
    return lo, hi


def exceedance_probability(pred: float, residuals: np.ndarray, threshold: float) -> float:
    """Share of past errors that would carry this forecast to ``threshold`` or above."""
    needed = np.log1p(threshold) - np.log1p(max(pred, 0.0))
    return float(np.mean(np.asarray(residuals) >= needed))


def _category(pred: np.ndarray) -> np.ndarray:
    # Left-closed bins: a forecast of exactly 5 is in [5, 20)
    return np.searchsorted(CATEGORY_EDGES, pred, side="right")


def past_residual_bands(frame: pd.DataFrame, coverage: float = 0.8, window: int | None = None,
                        mondrian: bool = False, threshold: float = 20.0,
                        min_calibration: int = 30) -> pd.DataFrame:
    """Band and exceedance probability for every row of a retrospective frame.

    ``frame`` needs ``date``, ``anchor_date``, ``actual_da`` and
    ``predicted_da`` (one row per forecast, any number of sites). Row i is
    calibrated by the rows whose ``date`` is on or before row i's
    ``anchor_date``. Returns ``band_lo``, ``band_hi``, ``p_exceed`` and
    ``n_calibration`` aligned to ``frame.index``; rows with fewer than
    ``min_calibration`` past errors get NaN.
    """
    dates = pd.to_datetime(frame["date"]).to_numpy()
    order = np.argsort(dates, kind="stable")
    cal_dates = dates[order]
    cal_resid = log_residuals(frame["actual_da"], frame["predicted_da"])[order]
    cal_cat = _category(np.clip(frame["predicted_da"].to_numpy(dtype=float), 0.0, None))[order]

    preds = np.clip(frame["predicted_da"].to_numpy(dtype=float), 0.0, None)
    anchors = pd.to_datetime(frame["anchor_date"]).to_numpy()
    n_known = np.searchsorted(cal_dates, anchors, side="right")
    cats = _category(preds)

    out = np.full((len(frame), 4), np.nan)
    for i, (pred, k, cat) in enumerate(zip(preds, n_known, cats)):
        resid = cal_resid[:k]
        if mondrian:
            same = resid[cal_cat[:k] == cat]
            if len(same) >= min_calibration:
                resid = same
        if window is not None:
            resid = resid[-window:]
        if len(resid) < min_calibration:
            continue
        lo, hi = band_from_residuals(pred, resid, coverage)
        out[i] = (lo, hi, exceedance_probability(pred, resid, threshold), len(resid))
    return pd.DataFrame(out, index=frame.index,
                        columns=["band_lo", "band_hi", "p_exceed", "n_calibration"])


def interval_score(actual, lo, hi, coverage: float) -> np.ndarray:
    """Winkler interval score on the log1p scale (lower is better): the band's
    width plus 2/alpha times how far the measurement falls outside it."""
    alpha = 1.0 - coverage
    y, a, b = (np.log1p(np.asarray(v, dtype=float)) for v in (actual, lo, hi))
    return (b - a) + (2 / alpha) * np.clip(a - y, 0, None) + (2 / alpha) * np.clip(y - b, 0, None)

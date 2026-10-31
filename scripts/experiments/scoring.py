"""Scoring helpers for the forecast experiments (docs/FORECAST_EXPERIMENTS.md).

Every comparison is paired: two prediction vectors on identical test rows,
one shared bootstrap index per replicate. Spike counting goes through
``forecasting.metrics`` (actual >= 20 is a spike, a prediction >= 12 alerts).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecasting.metrics import detection_scores, exceeds  # noqa: E402

SPIKE = 20.0
ALERT = 12.0
N_BOOT = 2000


def r2(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def transition_mask(df: pd.DataFrame, actual_col: str = "actual_da") -> np.ndarray:
    """Spikes whose previous measurement at the same site (in date order) was below 20.

    Same definition as docs/QUICKSTART_FOR_REVIEWERS.md; ``df`` must be sorted by date.
    """
    prev = df.groupby("site")[actual_col].shift(1)
    return ((prev < SPIKE) & (df[actual_col] >= SPIKE)).to_numpy()


def summarize(y, p, trans: np.ndarray | None = None, alert: float = ALERT) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    det = detection_scores(exceeds(y, SPIKE), exceeds(p, alert))
    out = {
        "n": int(len(y)),
        "r2": r2(y, p),
        "mae": float(np.mean(np.abs(y - p))),
        "bias": float(np.mean(p - y)),
        "recall": det["recall"],
        "precision": det["precision"],
        "f2": det["f2"],
    }
    if trans is not None and trans.any():
        out["trans_recall"] = float(exceeds(p[trans], alert).mean())
        out["n_trans"] = int(trans.sum())
    return out


def paired_delta_r2(y, p_base, p_new, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """R2(new) - R2(base) with a paired row-bootstrap 95% CI."""
    y = np.asarray(y, float)
    a = np.asarray(p_base, float)
    b = np.asarray(p_new, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = r2(y[idx], b[idx]) - r2(y[idx], a[idx])
    return r2(y, b) - r2(y, a), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def paired_delta_mae(y, p_base, p_new, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    y = np.asarray(y, float)
    ea = np.abs(y - np.asarray(p_base, float))
    eb = np.abs(y - np.asarray(p_new, float))
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.array([np.mean(eb[idx] - ea[idx]) for idx in (rng.integers(0, n, size=n) for _ in range(n_boot))])
    return float(np.mean(eb - ea)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def r2_ci(y, p, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    s = np.array([r2(y[idx], p[idx]) for idx in (rng.integers(0, n, size=n) for _ in range(n_boot))])
    return r2(y, p), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

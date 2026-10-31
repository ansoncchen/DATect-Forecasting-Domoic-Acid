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


# ---------------------------------------------------------------------------
# Onset skill: the first crossing of 20, scored only where it can happen
# ---------------------------------------------------------------------------
# A candidate row has its last known value at the anchor (the naive forecast) below 20;
# an onset is a candidate whose measurement is >= 20; the rest of the candidates are
# quiet weeks. Every score is a ranking of candidate rows (higher = more alarming).

def auc(pos: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted as 1/2."""
    from scipy.stats import rankdata
    pos = np.asarray(pos, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def expected_hits_top_k(score: np.ndarray, pos: np.ndarray, k: float) -> float:
    """Expected positives among the k highest scores, ties at the cut broken at random."""
    order = np.argsort(-score, kind="stable")
    s, p = score[order], pos[order].astype(float)
    k = int(round(k))
    if k <= 0:
        return 0.0
    if k >= len(s):
        return float(p.sum())
    cut = s[k - 1]
    above = s > cut
    tie = s == cut
    need = k - above.sum()
    return float(p[above].sum() + (p[tie].mean() * need if tie.any() else 0.0))


def onset_metrics(naive, actual, score, alert=None, fars=(0.10, 0.20)) -> dict:
    """Onset recall at fixed quiet-week false-alarm rates, AUC, and the gain over the
    last-value rule (rank candidates by the naive forecast) at the same number of alerts.

    ``alert`` (optional boolean array) adds the recall / false-alarm rate of a fixed
    operating point (e.g. the hybrid rule) and its matched last-value comparison.
    """
    naive = np.asarray(naive, float)
    actual = np.asarray(actual, float)
    score = np.asarray(score, float)
    cand = naive < SPIKE
    on = actual[cand] >= SPIKE
    s = score[cand]
    nv = naive[cand]
    quiet = s[~on]
    out = {"n_cand": int(cand.sum()), "n_onset": int(on.sum()), "onset_auc": auc(on, s),
           "naive_auc": auc(on, nv)}
    for far in fars:
        n_fa = int(np.floor(far * len(quiet)))
        thr = np.sort(quiet)[::-1][n_fa] if n_fa < len(quiet) else -np.inf
        fire = s > thr
        k = int(fire.sum())
        hits = int((fire & on).sum())
        triv = expected_hits_top_k(nv, on, k)
        tag = int(round(far * 100))
        out[f"onset_recall_far{tag}"] = hits / on.sum() if on.sum() else float("nan")
        out[f"onset_hits_far{tag}"] = hits
        out[f"onset_vs_lastvalue_far{tag}"] = hits - triv
    if alert is not None:
        a = np.asarray(alert, bool)[cand]
        out["alert_onset_recall"] = float((a & on).sum() / on.sum()) if on.sum() else float("nan")
        out["alert_quiet_far"] = float(a[~on].mean())
        out["alert_onset_hits"] = int((a & on).sum())
        out["alert_vs_lastvalue"] = int((a & on).sum()) - expected_hits_top_k(nv, on, a.sum())
    return out


def paired_delta_auc(naive, actual, s_base, s_new, n_boot: int = N_BOOT, seed: int = 0):
    """AUC(new) - AUC(base) on candidate rows, paired row bootstrap 95% CI."""
    naive = np.asarray(naive, float)
    cand = naive < SPIKE
    on = np.asarray(actual, float)[cand] >= SPIKE
    a, b = np.asarray(s_base, float)[cand], np.asarray(s_new, float)[cand]
    rng = np.random.default_rng(seed)
    n = len(on)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if on[idx].any() and (~on[idx]).any():
            d.append(auc(on[idx], b[idx]) - auc(on[idx], a[idx]))
    return auc(on, b) - auc(on, a), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def r2_ci(y, p, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    s = np.array([r2(y[idx], p[idx]) for idx in (rng.integers(0, n, size=n) for _ in range(n_boot))])
    return r2(y, p), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

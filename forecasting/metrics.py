"""
Shared evaluation metrics.

The evaluation scripts, the API summary and the engine's log line all score
spike detection through these helpers, so how a spike is counted lives in one
place.

Detection scores are computed from confusion counts with the same formulas
scikit-learn uses (e.g. F-beta = (1+b^2)*tp / (b^2*(tp+fn) + (tp+fp))), so they
match ``recall_score`` / ``precision_score`` / ``fbeta_score`` bit for bit.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def exceeds(values, threshold: float) -> np.ndarray:
    """Boolean mask of values at or above ``threshold``.

    Project-wide convention: reaching a threshold counts. A DA measurement of
    exactly 20 ug/g is a spike, and a 12 ug/g prediction fires the alert.
    """
    return np.asarray(values, dtype=float) >= threshold


def detection_scores(actual_event, predicted_event) -> dict:
    """Confusion counts plus recall, precision, F1, F2, FNR and FPR.

    Both inputs are boolean (or 0/1) arrays of equal length. Ratios with an
    empty denominator are reported as 0.0, matching ``zero_division=0``.
    """
    actual = np.asarray(actual_event, dtype=bool)
    predicted = np.asarray(predicted_event, dtype=bool)
    tp = int((actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    fp = int((~actual & predicted).sum())
    tn = int((~actual & ~predicted).sum())

    def _ratio(num, den):
        return num / den if den > 0 else 0.0

    recall = _ratio(tp, tp + fn)
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": recall,
        "precision": _ratio(tp, tp + fp),
        "f1": _ratio(2 * tp, 1 * (tp + fn) + (tp + fp)),
        "f2": _ratio(5 * tp, 4 * (tp + fn) + (tp + fp)),
        "fnr": 1.0 - recall,
        "fpr": _ratio(fp, fp + tn),
    }


def bootstrap_samples(
    arrays: Sequence[np.ndarray],
    statistic: Callable[..., float],
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Row-level bootstrap distribution of ``statistic(*arrays)``.

    Each replicate draws one index vector (``rng.integers(0, n, size=n)``) and
    applies it to every array, so paired statistics stay paired.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = statistic(*(a[idx] for a in arrays))
    return samples


def onset_scores(last_known, actual, alert=None, score=None, threshold: float = 20.0) -> dict:
    """How well spike *onsets* are caught: rows whose last value known at the
    anchor was below ``threshold`` and whose measurement reached it.

    The other below-threshold rows are the quiet weeks, the onsets' negatives.
    Rows already at or above the threshold (and rows with no known value) are
    left out, because carrying the last value forward catches those for free.
    ``alert`` (boolean per row) gives onset recall and the quiet-week alarm
    rate; ``score`` (e.g. the classifier probability) gives the onset AUC.
    """
    last = np.asarray(last_known, dtype=float)
    reached = exceeds(actual, threshold)
    below = ~np.isnan(last) & ~exceeds(last, threshold)
    onset, quiet = below & reached, below & ~reached
    out = {"n_onsets": int(onset.sum()), "n_quiet": int(quiet.sum())}
    if alert is not None:
        alert = np.asarray(alert, dtype=bool)
        out["onsets_caught"] = int((alert & onset).sum())
        out["onset_recall"] = out["onsets_caught"] / out["n_onsets"] if out["n_onsets"] else 0.0
        out["quiet_alarm_rate"] = float((alert & quiet).sum() / out["n_quiet"]) if out["n_quiet"] else 0.0
    if score is not None and onset.any() and quiet.any():
        from sklearn.metrics import roc_auc_score
        out["onset_auc"] = float(roc_auc_score(onset[below], np.asarray(score, dtype=float)[below]))
    return out

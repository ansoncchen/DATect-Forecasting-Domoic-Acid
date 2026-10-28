"""forecasting.metrics must reproduce scikit-learn's detection scores exactly.

The headline chronological numbers were computed with sklearn's
recall/precision/fbeta; the shared helper replaces those calls, so equality is
checked bit for bit (==, not approx) across random and edge-case inputs.
"""
import numpy as np
import pytest
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

from forecasting.metrics import bootstrap_samples, detection_scores, exceeds


def _sklearn_scores(actual, predicted):
    kw = {"zero_division": 0}
    return {
        "recall": float(recall_score(actual, predicted, **kw)),
        "precision": float(precision_score(actual, predicted, **kw)),
        "f1": float(f1_score(actual, predicted, **kw)),
        "f2": float(fbeta_score(actual, predicted, beta=2, **kw)),
    }


@pytest.mark.parametrize("seed", range(25))
def test_detection_scores_match_sklearn_exactly(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 400))
    actual = rng.random(n) < rng.uniform(0.02, 0.6)
    predicted = rng.random(n) < rng.uniform(0.02, 0.6)
    ours = detection_scores(actual, predicted)
    for key, value in _sklearn_scores(actual, predicted).items():
        assert ours[key] == value, (key, ours[key], value)


@pytest.mark.parametrize("actual,predicted", [
    ([0, 0, 0], [0, 0, 0]),   # no events, no alerts
    ([1, 1, 1], [1, 1, 1]),   # all events caught
    ([1, 0, 1], [0, 0, 0]),   # events, no alerts
    ([0, 0, 0], [1, 0, 1]),   # alerts, no events
])
def test_detection_scores_edge_cases(actual, predicted):
    ours = detection_scores(actual, predicted)
    for key, value in _sklearn_scores(actual, predicted).items():
        assert ours[key] == value, (key, ours[key], value)
    assert ours["tp"] + ours["fn"] + ours["fp"] + ours["tn"] == len(actual)


def test_exceeds_inclusive_vs_strict():
    values = [19.9, 20.0, 20.1]
    assert exceeds(values, 20.0, inclusive=True).tolist() == [False, True, True]
    assert exceeds(values, 20.0, inclusive=False).tolist() == [False, False, True]


def test_bootstrap_samples_keeps_pairs_and_rng_stream():
    a = np.arange(10.0)
    b = a * 2
    # paired resampling: statistic of (a, 2a) is always exactly 2
    ratios = bootstrap_samples((a + 1, b + 2), lambda x, y: float(np.mean(y / x)), 50,
                               np.random.default_rng(0))
    assert np.all(ratios == 2.0)
    # one integers(0, n, size=n) draw per replicate, in order
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    got = bootstrap_samples((a,), lambda x: float(x.sum()), 5, rng_a)
    want = [float(a[rng_b.integers(0, 10, size=10)].sum()) for _ in range(5)]
    assert got.tolist() == want

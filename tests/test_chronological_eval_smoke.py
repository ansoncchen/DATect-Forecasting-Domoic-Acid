"""
Smoke test for scripts/eval/chronological_eval.py.

We don't run the full retrospective in CI (too slow); instead we verify
that the script's deterministic-window contract holds: given a known
predictions parquet, the chronological eval produces the expected
window slicing + metric values.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


HOLDOUT_PARQUETS = [
    _REPO_ROOT / "results" / "chronological" / "chronological_regression_ensemble_20220101_20240101.parquet",
    _REPO_ROOT / "results" / "chronological" / "chronological_regression_ensemble_20190101_20220101.parquet",
]

CHRONO_JSON = (_REPO_ROOT / "results" / "chronological" /
               "chronological_regression_ensemble_20220101_20240101.json")


@pytest.mark.skipif(
    not all(p.exists() for p in HOLDOUT_PARQUETS),
    reason="Chronological eval artifacts not present (run scripts/eval/chronological_eval.py)",
)
def test_chronological_holdout_in_window():
    """Every row in the holdout parquet must fall in [2022-01-01, 2024-01-01)."""
    df = pd.read_parquet(HOLDOUT_PARQUETS[0])
    df["date"] = pd.to_datetime(df["date"])
    assert (df["date"] >= "2022-01-01").all(), "Some holdout rows before window start"
    assert (df["date"] < "2024-01-01").all(), "Some holdout rows past window end"


@pytest.mark.skipif(
    not CHRONO_JSON.exists(),
    reason="Chronological JSON not present",
)
def test_chronological_metrics_match_paper():
    """Headline holdout numbers must match docs/CORRECTED_NUMBERS.md.

    The chronological eval is deterministic (every real DA measurement in the
    window), so a re-run that changes these is a behavior change: update the
    docs, README and paper together with this test.
    """
    import json
    with open(CHRONO_JSON) as f:
        data = json.load(f)
    ov = data["overall"]
    assert ov["n"] == 409
    assert ov["n_spikes"] == 84
    assert ov["r2"][0] == pytest.approx(0.4734, abs=5e-4)
    assert ov["mae"][0] == pytest.approx(7.337, abs=5e-3)
    assert ov["spike_recall"] == pytest.approx(0.8452, abs=5e-4)
    assert ov["spike_f2"] == pytest.approx(0.7290, abs=5e-4)


@pytest.mark.skipif(
    not all(p.exists() for p in HOLDOUT_PARQUETS),
    reason="Chronological eval artifacts not present",
)
def test_chronological_eval_serves_the_random_forest():
    """The reported forecast is the random forest alone at every beach
    (docs/FORECAST_EXPERIMENTS.md §3c): the stored blend equals its RF component exactly."""
    for path in HOLDOUT_PARQUETS:
        df = pd.read_parquet(path)
        assert df["site"].nunique() >= 9
        assert (df["ensemble_prediction"] == df["predicted_da_rf"]).all(), path.name
        assert (df["predicted_da"] == df["predicted_da_rf"]).all(), path.name

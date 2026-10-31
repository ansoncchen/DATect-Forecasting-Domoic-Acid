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
    assert ov["r2"][0] == pytest.approx(0.497, abs=5e-4)
    assert ov["mae"][0] == pytest.approx(7.60, abs=5e-3)
    assert ov["spike_recall"] == pytest.approx(0.833, abs=5e-4)
    assert ov["spike_f2"] == pytest.approx(0.729, abs=5e-4)


@pytest.mark.skipif(
    not all(p.exists() for p in HOLDOUT_PARQUETS),
    reason="Chronological eval artifacts not present",
)
def test_chronological_eval_uses_grid_winner_config():
    """Holdout predictions must reflect each site's grid-winner ensemble weights.

    The parquet stores the blended prediction and the RF component, so RF-only
    sites must match the RF prediction exactly and XGB-only sites must not.
    """
    import numpy as np
    from forecasting.per_site_models import SITE_SPECIFIC_CONFIGS

    df = pd.read_parquet(HOLDOUT_PARQUETS[0])
    for site, cfg in SITE_SPECIFIC_CONFIGS.items():
        w_xgb, w_rf, _ = cfg["ensemble_weights"]
        sub = df[df["site"] == site]
        if sub.empty:
            continue
        matches_rf = np.isclose(sub["predicted_da"], sub["predicted_da_rf"]).mean()
        if (w_xgb, w_rf) == (0.0, 1.0):
            assert matches_rf == 1.0, f"{site} is RF-only but ensemble != RF prediction"
        elif (w_xgb, w_rf) == (1.0, 0.0):
            assert matches_rf < 0.5, f"{site} is XGB-only but ensemble tracks the RF prediction"

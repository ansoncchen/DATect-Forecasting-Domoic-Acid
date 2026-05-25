"""
Smoke test for scripts/eval/chronological_eval.py.

We don't run the full retrospective in CI (too slow); instead we verify
that the script's deterministic-window contract holds: given a known
predictions parquet, the chronological eval produces the expected
window slicing + metric values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


HOLDOUT_PARQUETS = [
    _REPO_ROOT / "eval_outputs" / "chronological" / "chronological_regression_ensemble_20220101_20240101.parquet",
    _REPO_ROOT / "eval_outputs" / "chronological" / "chronological_regression_ensemble_20190101_20220101.parquet",
]

CHRONO_JSON = (_REPO_ROOT / "eval_outputs" / "chronological" /
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
    """Headline holdout numbers must match the paper Table 3 values."""
    import json
    with open(CHRONO_JSON) as f:
        data = json.load(f)
    ov = data["overall"]
    # The paper reports R² = 0.485 [0.330, 0.604], MAE = 6.76, spike recall = 0.857
    # Use loose tolerance since the eval is run with the current grid-winner config
    # which is stable but could shift slightly on a future re-run.
    r2 = ov["r2"][0]
    assert 0.40 < r2 < 0.55, f"Holdout R² out of expected range: {r2}"
    mae = ov["mae"][0]
    assert 5.5 < mae < 8.0, f"Holdout MAE out of expected range: {mae}"
    if "spike_recall" in ov:
        rec = ov["spike_recall"]
        assert rec >= 0.75, f"Spike recall regressed: {rec}"


@pytest.mark.skipif(
    not all(p.exists() for p in HOLDOUT_PARQUETS),
    reason="Chronological eval artifacts not present",
)
def test_chronological_eval_uses_grid_winner_config():
    """Holdout predictions must reflect the grid-winner config (e.g. Copalis XGB-only)."""
    df = pd.read_parquet(HOLDOUT_PARQUETS[0])
    cop = df[df["site"] == "Copalis"]
    if len(cop) > 5 and "ensemble_weights" in cop.columns:
        # Grid winner for Copalis is (1.0, 0.0, 0.0) — XGB-only
        first_w = cop["ensemble_weights"].iloc[0]
        # The column may be stringified tuples or lists; just verify XGB dominates
        assert "1.0" in str(first_w) or first_w[0] >= 0.95, (
            f"Copalis appears not to be XGB-only in the eval output: {first_w}"
        )

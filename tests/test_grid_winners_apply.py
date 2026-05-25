"""
Regression test: grid-winner per-site configs from
config/tuned_hyperparameters.json must match the snapshot in the
grid_winners_overrides JSON format that the leak-test sbatch produces.

This guards against three failure modes:
  1. Someone edits per_site_models.py or the JSON manually and breaks
     consistency with the grid-search-winner snapshot.
  2. The JSON loader regresses (e.g. fails to convert list→tuple for
     ensemble_weights, which would silently break unpacking).
  3. Future grid-search reruns produce numerically inconsistent overrides
     that don't actually apply when loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# Expected per-site grid-winner configs as of the 2026-05-23 leak-free CV.
# These came directly from eval_outputs/grid_search_results/*_summary.json
# and were promoted to config/tuned_hyperparameters.json in commit cdc2a32c.
# If a future grid search produces different winners, update both this
# fixture AND the JSON in the same commit.
EXPECTED_GRID_WINNERS = {
    "Cannon Beach":  {"ensemble_weights": (0.00, 1.00, 0.00), "prediction_clip_q": 0.95, "prediction_clip_max": None},
    "Clatsop Beach": {"ensemble_weights": (1.00, 0.00, 0.00), "prediction_clip_q": 0.95, "prediction_clip_max": None},
    "Coos Bay":      {"ensemble_weights": (0.00, 1.00, 0.00), "prediction_clip_q": 0.95, "prediction_clip_max": None},
    "Copalis":       {"ensemble_weights": (1.00, 0.00, 0.00), "prediction_clip_q": 0.97, "prediction_clip_max": None},
    "Gold Beach":    {"ensemble_weights": (1.00, 0.00, 0.00), "prediction_clip_q": 0.97, "prediction_clip_max": None},
    "Kalaloch":      {"ensemble_weights": (0.00, 1.00, 0.00), "prediction_clip_q": 0.95, "prediction_clip_max": None},
    "Long Beach":    {"ensemble_weights": (1.00, 0.00, 0.00), "prediction_clip_q": 0.95, "prediction_clip_max": None},
    "Newport":       {"ensemble_weights": (1.00, 0.00, 0.00), "prediction_clip_q": 0.95, "prediction_clip_max": None},
    "Quinault":      {"ensemble_weights": (1.00, 0.00, 0.00), "prediction_clip_q": 0.95, "prediction_clip_max": None},
    "Twin Harbors":  {"ensemble_weights": (0.25, 0.75, 0.00), "prediction_clip_q": 0.97, "prediction_clip_max": None},
}


def test_grid_winners_match_loaded_json():
    """JSON must contain the snapshot of grid-winner overrides."""
    from forecasting.tuned_config import get_per_site
    sites = get_per_site()
    for site, expected in EXPECTED_GRID_WINNERS.items():
        actual = sites[site]
        for key, exp_val in expected.items():
            act_val = actual[key]
            assert act_val == exp_val, (
                f"{site}.{key}: expected {exp_val}, got {act_val}. "
                "If grid search has been re-run, update EXPECTED_GRID_WINNERS in this test."
            )


def test_grid_winners_apply_to_runtime_dict():
    """per_site_models.SITE_SPECIFIC_CONFIGS must reflect the JSON winners."""
    # Force reload to bypass any cached state
    for mod in list(sys.modules):
        if "per_site_models" in mod or "tuned_config" in mod:
            del sys.modules[mod]
    from forecasting.per_site_models import SITE_SPECIFIC_CONFIGS
    for site, expected in EXPECTED_GRID_WINNERS.items():
        cfg = SITE_SPECIFIC_CONFIGS[site]
        for key, exp_val in expected.items():
            act_val = cfg[key]
            assert act_val == exp_val, (
                f"Runtime {site}.{key}: expected {exp_val}, got {act_val}. "
                "Likely cause: per_site_models.py was hand-edited and drifted "
                "from config/tuned_hyperparameters.json."
            )


def test_ensemble_weights_are_tuples_after_load():
    """Critical: ensemble_weights must be tuple, not list, for unpacking."""
    from forecasting.tuned_config import get_per_site
    for site, cfg in get_per_site().items():
        w = cfg["ensemble_weights"]
        # Must support tuple indexing (w[0], w[1], w[2])
        assert isinstance(w, tuple), f"{site}: weights are {type(w).__name__}"
        # Spot check unpacking works
        w_xgb, w_rf, w_naive = w
        assert isinstance(w_xgb, (int, float))

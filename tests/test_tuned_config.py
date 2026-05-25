"""
Regression tests for forecasting/tuned_config.py — the JSON loader that
backs the single-source-of-truth refactor (2026-05-23).

Any future change that breaks these contracts will surface immediately
rather than silently corrupting per-site model behavior.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------
# JSON file existence + schema
# --------------------------------------------------------------------------

def test_json_file_exists():
    """Canonical JSON must live at config/tuned_hyperparameters.json."""
    p = _REPO_ROOT / "config" / "tuned_hyperparameters.json"
    assert p.exists(), f"Missing canonical JSON at {p}"


def test_json_top_level_schema():
    """JSON must contain the expected top-level structural keys."""
    from forecasting.tuned_config import load_tuned
    data = load_tuned()
    for key in ("version", "schema_version", "tuning_protocol",
                "eval_windows", "provenance", "global", "per_site"):
        assert key in data, f"Missing top-level key: {key!r}"


def test_eval_windows_schema():
    """eval_windows must define validation and holdout boundaries."""
    from forecasting.tuned_config import get_eval_windows
    w = get_eval_windows()
    for key in ("validation_start", "validation_end",
                "holdout_start", "holdout_end"):
        assert key in w, f"Missing eval_windows key: {key!r}"
    # Sanity: ISO date format
    import pandas as pd
    for date_str in (w["validation_start"], w["validation_end"],
                     w["holdout_start"], w["holdout_end"]):
        pd.Timestamp(date_str)  # raises if malformed


def test_global_block_required_keys():
    """global block must contain every tunable consumed by config.py."""
    from forecasting.tuned_config import get_global
    g = get_global()
    required = (
        "xgb_base_params", "rf_base_params", "rf_conservative_params",
        "spike_classifier_params", "param_grid",
        "min_training_for_tuning", "calibration_fraction",
        "max_calibration_rows", "min_tuning_samples",
        "spike_alert_prob_threshold", "spike_regression_alert_threshold",
        "spike_threshold_actual", "linear_regression_alpha",
        "prediction_clip_q_default", "zero_importance_features",
    )
    missing = [k for k in required if k not in g]
    assert not missing, f"global block missing keys: {missing}"


# --------------------------------------------------------------------------
# Per-site contract
# --------------------------------------------------------------------------

EXPECTED_SITES = {
    "Cannon Beach", "Clatsop Beach", "Coos Bay", "Copalis", "Gold Beach",
    "Kalaloch", "Long Beach", "Newport", "Quinault", "Twin Harbors",
}


def test_per_site_has_all_10_sites():
    """JSON must define configs for exactly the 10 PNW monitoring sites."""
    from forecasting.tuned_config import get_per_site
    sites = set(get_per_site().keys())
    assert sites == EXPECTED_SITES, (
        f"Site mismatch — got {sites}, expected {EXPECTED_SITES}"
    )


def test_per_site_required_fields():
    """Every per-site config must contain the standard 7 fields."""
    from forecasting.tuned_config import get_per_site
    required = ("xgb_params", "rf_params", "param_grid", "feature_subset",
                "ensemble_weights", "prediction_clip_q", "prediction_clip_max")
    for site, cfg in get_per_site().items():
        missing = [k for k in required if k not in cfg]
        assert not missing, f"{site} missing keys: {missing}"


def test_ensemble_weights_normalized():
    """Each site's ensemble_weights must sum to 1.0 (XGB + RF + naive)."""
    from forecasting.tuned_config import get_per_site
    for site, cfg in get_per_site().items():
        w = cfg["ensemble_weights"]
        assert len(w) == 3, f"{site}: expected 3-tuple weights, got {len(w)}"
        s = sum(w)
        assert abs(s - 1.0) < 1e-6, f"{site}: weights sum to {s}, not 1.0"


def test_ensemble_weights_returned_as_tuple():
    """Runtime expects tuple (not list) for unpacking via index access."""
    from forecasting.tuned_config import get_per_site
    for site, cfg in get_per_site().items():
        w = cfg["ensemble_weights"]
        assert isinstance(w, tuple), (
            f"{site}: ensemble_weights is {type(w).__name__}, expected tuple "
            "(loader must convert list→tuple at load time per CLAUDE.md gotcha)"
        )


def test_prediction_clip_q_in_valid_range():
    """clip_q must be a probability in (0, 1] or None."""
    from forecasting.tuned_config import get_per_site
    for site, cfg in get_per_site().items():
        q = cfg.get("prediction_clip_q")
        if q is None:
            continue
        assert 0.0 < q <= 1.0, f"{site}: clip_q = {q} out of (0, 1]"


# --------------------------------------------------------------------------
# Cache + reproducibility
# --------------------------------------------------------------------------

def test_load_tuned_is_cached():
    """load_tuned() uses lru_cache; repeated calls return the same dict."""
    from forecasting.tuned_config import load_tuned
    first = load_tuned()
    second = load_tuned()
    assert first is second, (
        "load_tuned() should return cached object on repeat calls "
        "(lru_cache contract)"
    )


def test_config_py_loads_from_json():
    """config.py reads global tunables from the JSON, not hardcoded."""
    # Force-reload to make sure we get a clean import
    for mod in list(sys.modules):
        if mod == "config" or "tuned_config" in mod:
            del sys.modules[mod]
    import config
    from forecasting.tuned_config import get_global
    g = get_global()
    # config.py value must equal JSON value
    assert config.SPIKE_THRESHOLD == float(g["spike_threshold_actual"])
    assert config.SPIKE_ALERT_PROB_THRESHOLD == float(g["spike_alert_prob_threshold"])
    assert config.MIN_TRAINING_FOR_TUNING == int(g["min_training_for_tuning"])
    assert config.CALIBRATION_FRACTION == float(g["calibration_fraction"])
    assert list(config.PARAM_GRID) == list(g["param_grid"])
    assert list(config.ZERO_IMPORTANCE_FEATURES) == list(g["zero_importance_features"])


def test_per_site_models_loads_from_json():
    """per_site_models.SITE_SPECIFIC_CONFIGS comes from the JSON loader."""
    for mod in list(sys.modules):
        if "per_site_models" in mod or "tuned_config" in mod:
            del sys.modules[mod]
    from forecasting.per_site_models import SITE_SPECIFIC_CONFIGS
    from forecasting.tuned_config import get_per_site
    json_sites = get_per_site()
    for site in EXPECTED_SITES:
        assert site in SITE_SPECIFIC_CONFIGS, f"{site} not in SITE_SPECIFIC_CONFIGS"
        # ensemble_weights must round-trip exactly
        assert SITE_SPECIFIC_CONFIGS[site]["ensemble_weights"] == json_sites[site]["ensemble_weights"]


# --------------------------------------------------------------------------
# Env-var overlay behavior
# --------------------------------------------------------------------------

def test_spike_classifier_env_var_overlay(tmp_path):
    """DATECT_SPIKE_CLASSIFIER_JSON should override JSON-loaded params."""
    # Write a temp override file
    override_path = tmp_path / "spike_override.json"
    override_path.write_text(json.dumps({"max_depth": 99, "n_estimators": 999}))
    os.environ["DATECT_SPIKE_CLASSIFIER_JSON"] = str(override_path)
    try:
        # Force re-import
        for mod in list(sys.modules):
            if mod == "config" or "tuned_config" in mod:
                del sys.modules[mod]
        import config
        assert config.SPIKE_CLASSIFIER_PARAMS["max_depth"] == 99
        assert config.SPIKE_CLASSIFIER_PARAMS["n_estimators"] == 999
    finally:
        del os.environ["DATECT_SPIKE_CLASSIFIER_JSON"]
        # Clear cache so subsequent tests get clean values
        for mod in list(sys.modules):
            if mod == "config" or "tuned_config" in mod:
                del sys.modules[mod]

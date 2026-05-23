"""Single source of truth for tuned hyperparameters.

Loads ``config/tuned_hyperparameters.json`` at import time. Both ``config.py``
and ``forecasting/per_site_models.py`` import from here instead of holding
hardcoded values, so the entire model configuration can be refreshed by
re-running ``scripts/tune/tune_all.py`` (TBD) and committing the new JSON.

Env-var overlays (``DATECT_HPARAM_OVERRIDE_JSON``, ``DATECT_SPIKE_CLASSIFIER_JSON``)
are still applied on top of the JSON values by the respective consumers.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PATH = _REPO_ROOT / "config" / "tuned_hyperparameters.json"


@lru_cache(maxsize=1)
def load_tuned(path: str | Path | None = None) -> Dict[str, Any]:
    """Load tuned_hyperparameters.json. Cached per-process (single read)."""
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Tuned hyperparameters JSON not found at {p}. "
            "Run scripts/tune/tune_all.py to generate it, or restore from git."
        )
    with open(p) as f:
        data = json.load(f)
    # Normalize: ensemble_weights stored as JSON list, runtime expects tuple
    for site_cfg in data.get("per_site", {}).values():
        if "ensemble_weights" in site_cfg and isinstance(site_cfg["ensemble_weights"], list):
            site_cfg["ensemble_weights"] = tuple(site_cfg["ensemble_weights"])
    return data


def get_per_site() -> Dict[str, Dict[str, Any]]:
    """Per-site config dict — keyed by site name."""
    return load_tuned()["per_site"]


def get_global() -> Dict[str, Any]:
    """Global tunable config block (XGB defaults, spike thresholds, etc.)."""
    return load_tuned()["global"]


def get_provenance() -> Dict[str, str]:
    """How each tunable group was selected — for paper Methods sections."""
    return load_tuned().get("provenance", {})


def get_eval_windows() -> Dict[str, str]:
    """Canonical train/val/test chronological split dates."""
    return load_tuned()["eval_windows"]

"""
Unit tests for backend/api.py endpoint robustness and JSON serialization.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.api import app, clean_for_json, resolve_site


def test_clean_for_json_numpy_and_math():
    """Verify clean_for_json converts numpy types and handles inf/nan properly."""
    data = {
        "np_float": np.float64(3.14),
        "np_int": np.int64(42),
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        "nested_dict": {
            "val": np.float32(1.23)
        },
        "list_val": [np.float64(1.0), float("nan")]
    }

    cleaned = clean_for_json(data)
    assert cleaned["np_float"] == pytest.approx(3.14)
    assert cleaned["np_int"] == 42
    assert cleaned["nan_val"] is None
    assert cleaned["inf_val"] is None
    assert cleaned["nested_dict"]["val"] == pytest.approx(1.23)
    assert cleaned["list_val"] == [1.0, None]


def test_clean_for_json_tuple():
    """Verify clean_for_json handles tuples by converting them to lists so they are JSON-serializable."""
    data = {
        "tuple_val": (np.float64(0.56), np.float64(0.44), 0.0),
        "nested_tuple": (1, (2, 3))
    }

    cleaned = clean_for_json(data)
    assert isinstance(cleaned["tuple_val"], list)
    assert cleaned["tuple_val"] == [0.56, 0.44, 0.0]
    assert cleaned["nested_tuple"] == [1, [2, 3]]


def test_resolve_site_mapping():
    """Verify resolve_site correctly resolves lowercase/slugified names to canonical ones."""
    assert resolve_site("copalis") == "Copalis"
    assert resolve_site("long-beach") == "Long Beach"
    assert resolve_site("twin-harbors") == "Twin Harbors"
    assert resolve_site("Unknown Site") == "Unknown Site"  # returns input as fallback


def test_api_root_and_health():
    """Verify basic health check and root endpoints return status 200."""
    with TestClient(app) as client:
        # Test Root
        response = client.get("/api")
        assert response.status_code == 200
        assert "DATect" in response.json()["message"]

        # Test Health
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_api_sites_endpoint():
    """Verify get_sites endpoint returns 10 Pacific Coast monitoring sites."""
    with TestClient(app) as client:
        response = client.get("/api/sites")
        assert response.status_code == 200
        data = response.json()
        assert "sites" in data
        assert "Copalis" in data["sites"]
        assert len(data["sites"]) == 10
        assert "min" in data["date_range"]
        assert "max" in data["date_range"]


def test_api_models_endpoint():
    """Verify get_models endpoint returns available models and descriptions."""
    with TestClient(app) as client:
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        assert "ensemble" in data["available_models"]["regression"]
        assert "descriptions" in data

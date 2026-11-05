"""
Unit tests for backend/api.py endpoint robustness and JSON serialization.
"""

from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parent.parent

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


def test_settings_travel_with_each_request():
    """The server holds no per-user settings: no config mutation endpoint, and
    /api/retrospective validates the task it is given."""
    with TestClient(app) as client:
        assert client.post("/api/config", json={"forecast_model": "naive"}).status_code == 405
        assert client.get("/api/config").status_code == 200
        response = client.post("/api/retrospective", json={"task": "bogus", "model": "ensemble"})
        assert response.status_code == 400


def test_metrics_summary_reads_committed_results():
    """/api/metrics/summary serves the holdout numbers from results/, not hardcoded ones."""
    with TestClient(app) as client:
        response = client.get("/api/metrics/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["holdout_window"]["start"] == "2022-01-01"
        assert len(data["overall"]["r2"]) == 3  # point estimate and bootstrap CI
        assert data["overall"]["n"] > 0
        assert data["per_site"]
        assert data["multi_seed"]["n_seeds"] == 5


def test_config_exposes_protocol_constants():
    """Thresholds, alert levels, categories, windows and site order come from the API."""
    import config
    from forecasting.raw_data_forecaster import load_raw_da_measurements
    with TestClient(app) as client:
        c = client.get("/api/config").json()
        sites = client.get("/api/sites").json()
    assert c["spike_threshold"] == config.SPIKE_THRESHOLD
    assert c["regression_alert_threshold"] == config.SPIKE_REGRESSION_ALERT_THRESHOLD
    assert c["classifier_alert_cutoff"] == config.SPIKE_ALERT_PROB_THRESHOLD
    assert c["band_level"] == 0.8
    levels = {lv["name"]: lv for lv in c["alert_levels"]}
    assert list(levels) == ["Quiet", "Watch", "Warning"]
    assert levels["Warning"]["placeholder"] is False and levels["Warning"]["cutoff"] == 0.35
    assert levels["Watch"]["placeholder"] is False
    assert c["category_bins"] == [0, 5, 20, 40, None] and len(c["category_labels"]) == 4
    assert c["eval_windows"]["holdout_start"] == "2022-01-01"
    assert c["sites_north_to_south"][0] == "Kalaloch" and c["sites_north_to_south"][-1] == "Gold Beach"
    assert len(c["sites_north_to_south"]) == 10
    newest = load_raw_da_measurements()["date"].max().date().isoformat()
    assert c["newest_measurement_date"] == newest
    assert c["latest_forecast_week"] == sites["date_range"]["max"]
    assert c["forecast_model"] == config.FORECAST_MODEL   # existing keys stay


def test_summary_reliability_per_site():
    """/api/metrics/summary adds per-site band coverage and alert counts beside its existing blocks."""
    with TestClient(app) as client:
        data = client.get("/api/metrics/summary").json()
    rel = data["reliability"]
    assert rel["holdout"]["per_site"]["Gold Beach"]["band"]["coverage"] == pytest.approx(0.375)
    assert rel["holdout"]["per_site"]["Newport"]["band"]["coverage"] == pytest.approx(0.479, abs=5e-4)
    assert rel["holdout"]["summary"]["onsets"]["caught"] == 19
    assert rel["validation"]["summary"]["onsets"]["caught"] == 27
    assert data["source"] and data["per_site"]["Gold Beach"]["spike_recall"] is not None


def test_region_rollup_puts_clatsop_and_cannon_in_oregon():
    import config
    from backend.api import _compute_summary
    assert config.SITE_STATES["Clatsop Beach"] == "OR" and config.SITE_STATES["Cannon Beach"] == "OR"
    rows = [{"site": s, "actual_da": float(i), "predicted_da": float(i) + 0.5}
            for s in ("Clatsop Beach", "Cannon Beach") for i in range(5)]
    regions = _compute_summary(rows, task="regression")["per_region"]
    assert regions["OR"]["n_sites"] == 2 and "WA" not in regions


def test_historical_returns_raw_measurements():
    """/api/historical serves the real measurements (one per site-day at the max), not the gap-filled weekly series."""
    from forecasting.raw_data_forecaster import load_raw_da_measurements
    m = load_raw_da_measurements()
    expected = m[m["site"] == "Newport"].sort_values("date")
    with TestClient(app) as client:
        body = client.get("/api/historical/newport", params={"limit": 100000}).json()
        assert client.get("/api/historical/atlantis").status_code == 404
    assert body["site"] == "Newport" and "raw measurements" in body["series"]
    assert body["rows"] == [{"date": d.date().isoformat(), "value": float(v)}
                            for d, v in zip(expected["date"], expected["da_raw"])]


def test_metrics_summary_is_computed_once_per_process():
    from backend import api
    assert hasattr(api._load_metrics_summary, "cache_info")


def test_site_map_reports_measurements_not_a_risk_verdict():
    """The map shows the last reading, which can be months old: no 'risk level', no reassuring green."""
    with TestClient(app) as client:
        plot = client.get("/api/visualizations/map").json()["plot"]
    title = plot["layout"]["title"]["text"]
    assert "Risk" not in title and "last measured" in title.lower()
    trace = plot["data"][0]
    assert not {"green", "yellow"} & set(trace["marker"]["color"])
    hover = " ".join(trace["hovertemplate"])
    assert "Risk Level" not in hover and "Last measured DA" in hover

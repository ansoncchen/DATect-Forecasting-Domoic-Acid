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


def test_settings_travel_with_each_request():
    """The server holds no per-user settings: no config mutation endpoint."""
    with TestClient(app) as client:
        assert client.post("/api/config", json={"forecast_model": "naive"}).status_code == 405
        assert client.get("/api/config").status_code == 200


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


def test_site_states_follow_the_columbia_border():
    import config
    assert config.SITE_STATES["Clatsop Beach"] == "OR" and config.SITE_STATES["Cannon Beach"] == "OR"
    assert config.SITE_STATES["Copalis"] == "WA" and config.SITE_STATES["Newport"] == "OR"


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


def _map_hovers():
    with TestClient(app) as client:
        trace = client.get("/api/visualizations/map").json()["plot"]["data"][0]
    return dict(zip(trace["text"], zip(trace["hovertemplate"], trace["marker"]["color"])))


_NO_MAPBOX_WARNING = pytest.mark.filterwarnings("ignore:.*scattermapbox.*:DeprecationWarning")


@_NO_MAPBOX_WARNING
def test_map_uses_the_canonical_measurements(monkeypatch):
    import pandas as pd

    import config
    from backend import api

    rows, expected = [], {}
    for k, site in enumerate(config.SITES):
        latest = pd.Timestamp("2023-06-01") + pd.Timedelta(days=7 * k)
        value = 1.25 + k
        # out of date order; the older row of the first site has the larger value
        rows += [(site, latest, value),
                 (site, latest - pd.Timedelta(days=30), value + 50.0),
                 (site, latest - pd.Timedelta(days=60), 0.5)]
        expected[site] = (value, latest.strftime("%Y-%m-%d"))
    frame = pd.DataFrame(rows[::-1], columns=["site", "date", "da_raw"])
    monkeypatch.setattr(api, "_raw_measurements", lambda: frame)
    hovers = _map_hovers()
    assert set(hovers) == set(expected)
    for site, (value, day) in expected.items():
        assert f"{value:.2f}" in hovers[site][0] and day in hovers[site][0]


@_NO_MAPBOX_WARNING
def test_map_degrades_to_no_data_when_no_raw_file_loads(monkeypatch):
    from backend import api

    def boom():
        raise ValueError("No raw DA measurements could be loaded.")
    monkeypatch.setattr(api, "_raw_measurements", boom)
    hovers = _map_hovers()
    assert len(hovers) == 10
    for hover, color in hovers.values():
        assert "no data" in hover and color == "#d0d5db"


@_NO_MAPBOX_WARNING
def test_map_marks_a_site_without_measurements(monkeypatch):
    from backend import api
    m = api._raw_measurements()
    monkeypatch.setattr(api, "_raw_measurements", lambda: m[m["site"] != "Newport"])
    hover, color = _map_hovers()["Newport"]
    assert "no data" in hover and color == "#d0d5db"

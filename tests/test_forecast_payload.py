"""The dashboard's forecast: one regression run, its conformal band, the alert's
parts and the measurements the forecast could see (dashboard design, section 6)."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import band
from backend import precomputed as pf
from forecasting.raw_data_forecaster import load_raw_da_measurements


class _Engine:
    """generate_single_forecast stand-in: records calls, returns a fixed forecast."""

    def __init__(self, predicted=5.3, spike_probability=0.02, result=True):
        self.calls, self.predicted, self.prob, self.result = [], predicted, spike_probability, result

    def generate_single_forecast(self, data_path, forecast_date, site, task, model_type):
        day = pd.Timestamp(forecast_date)
        self.calls.append((site, day.date().isoformat(), task, model_type))
        if not self.result:
            return None
        return {
            "forecast_date": day, "anchor_date": day - pd.Timedelta(days=7), "site": site,
            "task": task, "model_type": model_type, "training_samples": 600,
            "predicted_da": self.predicted, "feature_importance": {"da_lag_1": 0.4},
            "naive_prediction": 2.0, "xgb_prediction": self.predicted, "rf_prediction": self.predicted,
            "ensemble_prediction": self.predicted, "ensemble_weights": [0.5, 0.5, 0.0],
            "predicted_category": 1, "spike_probability": self.prob,
            "spike_alert": self.prob >= 0.10 or self.predicted >= 12,
        }


@pytest.fixture
def client(monkeypatch):
    """The API with no precomputed store, so every forecast reaches the stub engine."""
    from backend import api

    monkeypatch.delenv(pf.STORE_ENV, raising=False)
    for attr in ("_loaded_root", "_store", "_load_error"):
        monkeypatch.setattr(pf, attr, None)

    def use(engine):
        monkeypatch.setattr(api, "get_forecast_engine", lambda: engine)
        return engine

    with TestClient(api.app) as c:
        c.use = use
        yield c


def _post(client, site, day, model="ensemble"):
    return client.post("/api/forecast/enhanced", json={"date": day, "site": site, "model": model}).json()


def test_enhanced_forecast_calls_the_engine_once_and_returns_the_band(client):
    engine = client.use(_Engine(predicted=5.3))
    body = _post(client, "Kalaloch", "2019-06-03")
    assert [c[2] for c in engine.calls] == ["regression"]
    assert "classification" not in body
    expected = band.forecast_band(5.3, pd.Timestamp("2019-05-27"))
    assert body["anchor_date"] == "2019-05-27"
    assert body["band"] == expected and body["band_p_exceed"] == expected["p_exceed"]
    assert body["category_probabilities"] == expected["category_probabilities"]
    cat = body["graphs"]["category_range"]
    assert cat["source"] == "range" and cat["class_probabilities"] == expected["category_probabilities"]
    assert body["graphs"]["level_range"]["band"] == expected
    # The blocks the current dashboard reads stay where they were
    assert body["regression"]["training_samples"] == 600 and "feature_importance" in body["regression"]
    assert body["ensemble"]["ensemble_weights"] == [0.5, 0.5, 0.0]


def test_enhanced_forecast_reports_alert_parts_and_level(client):
    client.use(_Engine(predicted=3.0, spike_probability=0.5))
    alert = _post(client, "Kalaloch", "2019-06-03")["alert"]   # last measurement 2.0
    assert alert["level"] == "Warning" and alert["level_reason"] is None
    assert alert["by_classifier"] is True and alert["by_regression"] is False and alert["alert"] is True


def test_no_level_once_the_last_measurement_reached_the_threshold(client):
    client.use(_Engine(predicted=25.0, spike_probability=0.9))
    m = load_raw_da_measurements()
    hit = m[(m.site == "Copalis") & (m.date >= "2019-01-01") & (m.da_raw >= 20)].iloc[0]
    later = m[(m.site == "Copalis") & (m.date > hit.date) & (m.date <= hit.date + pd.Timedelta(days=7))]
    assert later.empty   # the >= 20 sample is the last one the forecast can see
    body = _post(client, "Copalis", (hit.date + pd.Timedelta(days=7)).date().isoformat())
    assert body["last_measurement"]["value"] == hit.da_raw
    assert body["alert"]["level"] is None
    assert body["alert"]["level_reason"] == "already_at_or_above_threshold"
    assert body["alert"]["alert"] is True


def test_recent_measurements_stop_at_the_anchor(client):
    client.use(_Engine())
    body = _post(client, "Newport", "2019-06-03")
    m = load_raw_da_measurements()
    seen = m[(m.site == "Newport") & (m.date <= "2019-05-27")].sort_values("date").tail(6)
    assert body["recent_measurements"] == [
        {"date": d.date().isoformat(), "value": float(v)} for d, v in zip(seen.date, seen.da_raw)]
    assert body["last_measurement"] == body["recent_measurements"][-1]


def test_no_forecast_is_reported_not_raised(client):
    client.use(_Engine(result=False))
    body = _post(client, "Kalaloch", "2019-06-03")
    assert body["success"] is False and "error" in body


def test_no_band_before_enough_past_errors(client):
    client.use(_Engine(predicted=3.0))
    body = _post(client, "Kalaloch", "2004-06-07")
    assert body["success"] is True and body["band"] is None
    assert body["band_p_exceed"] is None and body["category_probabilities"] is None
    assert "level_range" not in body["graphs"] and "category_range" not in body["graphs"]


def test_all_sites_returns_every_site_for_one_week(client):
    engine = client.use(_Engine(predicted=4.0))
    config_body = client.get("/api/config").json()
    body = client.get("/api/forecast/all").json()
    assert body["date"] == config_body["latest_forecast_week"]
    assert [s["site"] for s in body["sites"]] == config_body["sites_north_to_south"]
    assert len(engine.calls) == 10 and {c[2] for c in engine.calls} == {"regression"}
    first = body["sites"][0]
    assert set(first) == {"site", "anchor_date", "predicted_da", "band", "alert", "last_measurement",
                          "source", "error"}
    assert first["alert"]["level"] in ("Quiet", "Watch", "Warning", None)
    picked = client.get("/api/forecast/all", params={"date": "2019-06-03"}).json()
    assert picked["date"] == "2019-06-03" and picked["sites"][0]["anchor_date"] == "2019-05-27"


def test_all_sites_rejects_a_malformed_date(client):
    client.use(_Engine())
    assert client.get("/api/forecast/all", params={"date": "next tuesday"}).status_code == 422



class _FailsAt(_Engine):
    """Raises for one site, as an engine error would."""

    def __init__(self, site, **kw):
        super().__init__(**kw)
        self.site = site

    def generate_single_forecast(self, data_path, forecast_date, site, task, model_type):
        if site == self.site:
            raise RuntimeError("engine failed")
        return super().generate_single_forecast(data_path, forecast_date, site, task, model_type)


def test_band_only_for_the_ensemble(client):
    """The band was calibrated on the ensemble's errors; other models get none."""
    client.use(_Engine(predicted=5.3))
    linear = _post(client, "Kalaloch", "2019-06-03", model="linear")
    assert linear["band"] is None and linear["band_reason"] == "calibrated for the ensemble only"
    assert linear["category_probabilities"] is None and "level_range" not in linear["graphs"]
    assert _post(client, "Kalaloch", "2019-06-03")["band_reason"] is None


def test_all_sites_rejects_unknown_models(client):
    client.use(_Engine())
    assert client.get("/api/forecast/all", params={"model": "bogus"}).status_code == 422


def test_all_sites_survives_one_failing_site(client):
    client.use(_FailsAt("Newport", predicted=4.0))
    response = client.get("/api/forecast/all", params={"date": "2019-06-03"})
    assert response.status_code == 200
    sites = {s["site"]: s for s in response.json()["sites"]}
    newport = sites.pop("Newport")
    assert newport["predicted_da"] is None and newport["error"]
    assert newport["last_measurement"] is not None and newport["anchor_date"] == "2019-05-27"
    assert all(s["predicted_da"] == 4.0 and s["error"] is None for s in sites.values())


def test_no_forecast_keeps_the_measurements_and_the_shape(client):
    client.use(_Engine(result=False))
    body = _post(client, "Kalaloch", "2019-06-03")
    assert body["success"] is False and body["anchor_date"] == "2019-05-27"
    assert body["last_measurement"] == body["recent_measurements"][-1]
    assert body["band"] is None and body["category_probabilities"] is None and body["alert"] is None


def test_dates_outside_the_supported_range_are_rejected(client):
    """Pandas cannot represent these dates; they are a bad request, not a server error."""
    client.use(_Engine())
    assert client.get("/api/forecast/all", params={"date": "0002-12-23"}).status_code == 422
    r = client.post("/api/forecast/enhanced", json={"date": "0002-12-23", "site": "Newport", "model": "ensemble"})
    assert r.status_code == 422
    assert client.get("/api/sites/Newport/history", params={"since": "0002-12-23"}).status_code == 422
    assert client.get("/api/sites/Newport/history", params={"until": "0002-12-23"}).status_code == 422
    assert client.get("/api/historical/newport", params={"start_date": "0002-12-23"}).status_code == 422


def test_a_future_date_says_it_is_past_the_data(client):
    client.use(_Engine(result=False))
    latest = client.get("/api/config").json()["latest_forecast_week"]
    body = _post(client, "Kalaloch", "2030-01-07")
    assert body["success"] is False
    assert "past the newest data" in body["error"] and latest in body["error"]
    assert "training data" not in body["error"]

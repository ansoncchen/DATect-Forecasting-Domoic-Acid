"""Serving precomputed forecasts from Delta (backend/precomputed_forecasts.py) and
the API's fallback to on-demand forecasts."""
import datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import precomputed_forecasts as pf


def _result(site="Newport", day="2019-06-03", task="regression"):
    """The shape generate_single_forecast returns (types as observed)."""
    return {
        "forecast_date": pd.Timestamp(day),
        "anchor_date": pd.Timestamp(day) - pd.Timedelta(days=7),
        "site": site, "task": task, "model_type": "ensemble", "training_samples": 855,
        "predicted_da": 5.32426643371582,
        "feature_importance": {"da_lag_1": np.float32(0.25), "sst": np.float32(1e-8)},
        "naive_prediction": 9.4, "xgb_prediction": 5.32426643371582,
        "rf_prediction": 7.277734477520231, "ensemble_prediction": 5.32426643371582,
        "ensemble_weights": [1.0, 0.0, 0.0],
        "bootstrap_quantiles": {"q05": 0.1, "q50": 4.0, "q95": 31.7},
        "predicted_category": 1, "spike_probability": 0.023508291691541672, "spike_alert": False,
    }


def test_codec_round_trips_exactly_with_types():
    r = _result()
    r.update(nan=float("nan"), np64=np.float64(0.1), npint=np.int64(3), npbool=np.bool_(True),
             tup=(1, 2.5), arr=np.array([0.5, np.nan], dtype=np.float32), none=None)
    back = pf.decode(pf.encode(r))
    assert pf.same(back, r)
    assert type(back["feature_importance"]["sst"]) is np.float32
    assert type(back["np64"]) is np.float64 and type(back["tup"]) is tuple
    assert pf.decode(pf.encode(None)) is None


def test_same_is_strict_about_values_and_types():
    assert not pf.same({"a": 1.0}, {"a": np.float64(1.0)})
    assert not pf.same({"a": np.float32(0.1)}, {"a": np.float32(0.1000001)})
    assert not pf.same([1, 2], (1, 2))
    assert pf.same({"a": float("nan")}, {"a": float("nan")})


def test_config_hash_is_stable_and_tracks_engine_variables(monkeypatch):
    first = pf.config_hash()
    assert pf.config_hash() == first
    monkeypatch.setenv("DATECT_USE_PER_SITE_MODELS", "false")
    assert pf.config_hash() != first
    monkeypatch.delenv("DATECT_USE_PER_SITE_MODELS")
    monkeypatch.setenv("DATECT_FORECAST_TABLE", "/elsewhere")  # serving, not the engine
    assert pf.config_hash() == first


@pytest.fixture
def table(tmp_path, monkeypatch):
    """A gold/forecasts table: Newport 2019-06-03 (both tasks), an insufficient-data
    entry, and a row from another config, which must not be served."""
    deltalake = pytest.importorskip("deltalake")
    current = pf.config_hash()
    rows = [("Newport", "2019-06-03", "regression", pf.encode(_result()), current),
            ("Newport", "2019-06-03", "classification", pf.encode(_result(task="classification")), current),
            ("Newport", "2003-01-13", "regression", pf.encode(None), current),
            ("Coos Bay", "2019-06-03", "regression", pf.encode(_result("Coos Bay")), "stale")]
    df = pd.DataFrame(rows, columns=["site", "day", "task", "result_json", "config_hash"])
    df["forecast_date"] = [datetime.date.fromisoformat(d) for d in df.pop("day")]
    df["model_type"] = "ensemble"
    df["model_version"] = "datect-forecaster/7"
    path = str(tmp_path / "forecasts")
    deltalake.write_deltalake(path, df)
    monkeypatch.setenv(pf.TABLE_ENV, path)
    monkeypatch.setattr(pf, "_loaded_uri", None)
    monkeypatch.setattr(pf, "_store", None)
    return path


def test_lookup_hits_misses_and_ignores_stale_rows(table):
    store = pf.load_from_env()
    assert len(store) == 3 and store.stale_rows == 1
    hit = pf.lookup("Newport", pd.Timestamp("2019-06-03"), "regression", "ensemble")
    assert hit.model_version == "datect-forecaster/7" and pf.same(hit.result, _result())
    assert pf.lookup("Newport", "2003-01-13", "regression", "ensemble") == pf.Hit(None, "datect-forecaster/7")
    assert pf.lookup("Newport", "2019-06-03", "regression", "naive") is None
    assert pf.lookup("Coos Bay", "2019-06-03", "regression", "ensemble") is None


def test_unreadable_table_falls_back_without_retrying(tmp_path, monkeypatch):
    monkeypatch.setenv(pf.TABLE_ENV, str(tmp_path / "missing"))
    monkeypatch.setattr(pf, "_loaded_uri", None)
    monkeypatch.setattr(pf, "_store", None)
    assert pf.lookup("Newport", "2019-06-03", "regression", "ensemble") is None
    assert pf._loaded_uri == str(tmp_path / "missing")  # remembered: no reload per request


class _Engine:
    def __init__(self):
        self.calls = []

    def generate_single_forecast(self, data_path, forecast_date, site, task, model_type):
        self.calls.append((site, pd.Timestamp(forecast_date).date().isoformat(), task, model_type))
        return _result(site, pd.Timestamp(forecast_date).date().isoformat(), task)


def test_enhanced_endpoint_serves_hits_and_computes_misses(table, monkeypatch):
    from backend import api

    engine = _Engine()
    monkeypatch.setattr(api, "get_forecast_engine", lambda: engine)
    body = {"date": "2019-06-03", "site": "Newport", "task": "regression", "model": "ensemble"}
    with TestClient(api.app) as client:
        served = client.post("/api/forecast/enhanced", json=body).json()
        assert engine.calls == [] and served["source"] == "precomputed"
        assert served["model_version"] == "datect-forecaster/7"

        monkeypatch.delenv(pf.TABLE_ENV)
        computed = client.post("/api/forecast/enhanced", json=body).json()
        assert len(engine.calls) == 2 and computed["source"] == "computed"
        assert computed["model_version"] is None

        monkeypatch.setenv(pf.TABLE_ENV, table)
        missed = client.post("/api/forecast/enhanced", json={**body, "model": "naive"}).json()
        assert missed["source"] == "computed" and len(engine.calls) == 4

    # Same forecast either way: the responses differ only in where they came from
    strip = ("source", "model_version")
    assert {k: v for k, v in served.items() if k not in strip} == \
        {k: v for k, v in computed.items() if k not in strip}


def test_forecast_endpoint_reports_its_source(table, monkeypatch):
    from backend import api

    engine = _Engine()
    monkeypatch.setattr(api, "get_forecast_engine", lambda: engine)
    with TestClient(api.app) as client:
        r = client.post("/api/forecast", json={"date": "2019-06-03", "site": "Newport",
                                               "task": "regression", "model": "ensemble"}).json()
    assert r["source"] == "precomputed" and r["prediction"] == 5.32426643371582 and engine.calls == []


def test_codec_fails_closed_on_types_it_cannot_round_trip():
    from collections import OrderedDict

    for value in (np.str_("x"), OrderedDict(a=1), np.zeros((0, 3)), np.array(1.5),
                  {"__ts__": "2019-06-03"}, {1: "int key"}):
        with pytest.raises(TypeError):
            pf.encode({"v": value})


def test_unreadable_row_is_a_miss_not_an_error(tmp_path, monkeypatch):
    deltalake = pytest.importorskip("deltalake")
    df = pd.DataFrame({"site": ["Newport"], "forecast_date": [datetime.date(2019, 6, 3)],
                       "task": ["regression"], "model_type": ["ensemble"],
                       "result_json": ['{"forecast_date": {"__ts__": "not a date"}}'],
                       "model_version": ["datect-forecaster/7"], "config_hash": [pf.config_hash()]})
    path = str(tmp_path / "forecasts")
    deltalake.write_deltalake(path, df)
    monkeypatch.setenv(pf.TABLE_ENV, path)
    monkeypatch.setattr(pf, "_loaded_uri", None)
    monkeypatch.setattr(pf, "_store", None)
    assert pf.lookup("Newport", "2019-06-03", "regression", "ensemble") is None


def test_stale_table_is_reported(table, caplog):
    with caplog.at_level("WARNING", logger=pf.__name__):
        store = pf.load_from_env()
    assert any("stale" in r.getMessage() for r in caplog.records if r.levelname == "WARNING")
    status = pf.status()
    assert status["entries"] == len(store) == 3 and status["stale_rows"] == 1
    assert status["expected_config_hash"] == pf.config_hash() and status["table_version"] == 0


def test_health_reports_the_table(table):
    from backend import api

    with TestClient(api.app) as client:
        health = client.get("/health").json()
    assert health["precomputed_forecasts"]["entries"] == 3


def test_health_without_a_table(monkeypatch):
    from backend import api

    monkeypatch.delenv(pf.TABLE_ENV, raising=False)
    with TestClient(api.app) as client:
        assert client.get("/health").json()["precomputed_forecasts"] is None

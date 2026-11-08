"""The precomputed store (backend/precomputed.py): Delta tables read with deltalake,
and the API's fallback to computing on demand."""
import datetime
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import precomputed as pf


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
        "predicted_category": 1, "spike_probability": 0.023508291691541672, "spike_alert": False,
    }


def _records(model="xgboost"):
    """/api/retrospective records (as retrospective_records builds them)."""
    days = pd.date_range("2019-01-07", periods=30, freq="7D")
    return [{"date": d.strftime("%Y-%m-%d"), "site": "Newport", "actual_da": float(i % 7),
             "predicted_da": float(i % 5) + (0.5 if model == "rf" else 0.0),
             "actual_category": 0, "predicted_category": 0, "anchor_date": d.strftime("%Y-%m-%d")}
            for i, d in enumerate(days)]


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


def test_codec_fails_closed_on_types_it_cannot_round_trip():
    from collections import OrderedDict

    for value in (np.str_("x"), OrderedDict(a=1), np.zeros((0, 3)), np.array(1.5),
                  {"__ts__": "2019-06-03"}, {1: "int key"}):
        with pytest.raises(TypeError):
            pf.encode({"v": value})


def test_config_hash_tracks_what_changes_a_row(monkeypatch, tmp_path):
    first = pf.config_hash()
    assert pf.config_hash() == first
    monkeypatch.setenv("DATECT_USE_PER_SITE_MODELS", "false")
    assert pf.config_hash() != first
    monkeypatch.delenv("DATECT_USE_PER_SITE_MODELS")
    # Serving and executor settings never change a stored row
    monkeypatch.setenv(pf.STORE_ENV, "/elsewhere")
    monkeypatch.setenv("DATECT_PARALLEL_BACKEND", "spark")
    assert pf.config_hash() == first
    # An override file counts by its contents, not only its path
    override = tmp_path / "rf.json"
    override.write_text('{"n_estimators": 10}')
    monkeypatch.setenv("DATECT_RF_PARAMS_JSON", str(override))
    with_file = pf.config_hash()
    override.write_text('{"n_estimators": 20}')
    assert pf.config_hash() != with_file


def _reset(monkeypatch):
    monkeypatch.setattr(pf, "_loaded_root", None)
    monkeypatch.setattr(pf, "_store", None)
    monkeypatch.setattr(pf, "_load_error", None)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A store directory with all four tables. Forecasts: Newport 2019-06-03 (both
    tasks), an insufficient-data entry, and a row from another config, which must not
    be served."""
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
    root = tmp_path / "gold"
    deltalake.write_deltalake(str(root / "forecasts"), df)
    deltalake.write_deltalake(str(root / "retrospective"), pd.DataFrame(
        [{"task": "regression", "model_type": m, "records_json": json.dumps(_records(m)),
          "config_hash": current} for m in ("xgboost", "rf")]))
    deltalake.write_deltalake(str(root / "spectral"), pd.DataFrame(
        [{"site": "Newport", "plots_json": json.dumps([{"data": [], "layout": {"title": "n"}}]),
          "config_hash": current},
         {"site": pf.ALL_SITES, "plots_json": json.dumps([{"data": [], "layout": {"title": "all"}}]),
          "config_hash": current}]))
    deltalake.write_deltalake(str(root / "correlation"), pd.DataFrame(
        [{"site": "Newport", "plot_json": json.dumps({"data": [{"z": [[1.0]]}]}), "config_hash": current}]))
    monkeypatch.setenv(pf.STORE_ENV, str(root))
    _reset(monkeypatch)
    return str(root)


def test_store_hits_misses_and_ignores_stale_rows(store):
    s = pf.load_from_env()
    assert s.tables["forecasts"]["entries"] == 3 and s.stale_rows == 1
    hit = pf.forecast("Newport", pd.Timestamp("2019-06-03"), "regression", "ensemble")
    assert hit.model_version == "datect-forecaster/7" and pf.same(hit.result, _result())
    assert pf.forecast("Newport", "2003-01-13", "regression", "ensemble") == pf.Hit(None, "datect-forecaster/7")
    assert pf.forecast("Newport", "2019-06-03", "regression", "naive") is None
    assert pf.forecast("Coos Bay", "2019-06-03", "regression", "ensemble") is None
    assert pf.retrospective("regression", "xgboost") == _records("xgboost")
    assert pf.retrospective("classification", "ensemble") is None
    assert pf.spectral(None)[0]["layout"]["title"] == "all"
    assert pf.correlation("Newport") == {"data": [{"z": [[1.0]]}]}
    assert pf.correlation("Coos Bay") is None


def test_a_store_without_some_tables_serves_the_rest(tmp_path, monkeypatch):
    deltalake = pytest.importorskip("deltalake")
    deltalake.write_deltalake(str(tmp_path / "correlation"), pd.DataFrame(
        [{"site": "Newport", "plot_json": "{}", "config_hash": pf.config_hash()}]))
    monkeypatch.setenv(pf.STORE_ENV, str(tmp_path))
    _reset(monkeypatch)
    assert pf.correlation("Newport") == {}
    assert pf.forecast("Newport", "2019-06-03", "regression", "ensemble") is None
    assert set(pf.status()["tables"]) == {"correlation"}


def test_unreadable_row_is_a_miss_not_an_error(tmp_path, monkeypatch):
    deltalake = pytest.importorskip("deltalake")
    df = pd.DataFrame({"site": ["Newport"], "forecast_date": [datetime.date(2019, 6, 3)],
                       "task": ["regression"], "model_type": ["ensemble"],
                       "result_json": ['{"forecast_date": {"__ts__": "not a date"}}'],
                       "model_version": ["datect-forecaster/7"], "config_hash": [pf.config_hash()]})
    deltalake.write_deltalake(str(tmp_path / "forecasts"), df)
    monkeypatch.setenv(pf.STORE_ENV, str(tmp_path))
    _reset(monkeypatch)
    assert pf.forecast("Newport", "2019-06-03", "regression", "ensemble") is None


def test_stale_store_is_reported(store, caplog):
    with caplog.at_level("WARNING", logger=pf.__name__):
        pf.load_from_env()
    assert any("stale" in r.getMessage() for r in caplog.records if r.levelname == "WARNING")
    status = pf.status()
    assert status["tables"]["forecasts"]["entries"] == 3
    assert status["tables"]["forecasts"]["stale_rows"] == 1
    assert status["expected_config_hash"] == pf.config_hash()


class _Engine:
    def __init__(self):
        self.calls = []

    def generate_single_forecast(self, data_path, forecast_date, site, task, model_type):
        self.calls.append((site, pd.Timestamp(forecast_date).date().isoformat(), task, model_type))
        return _result(site, pd.Timestamp(forecast_date).date().isoformat(), task)

    def run_retrospective_evaluation(self, task, model_type):
        self.calls.append(("retrospective", task, model_type))
        return pd.DataFrame({"date": pd.to_datetime(["2019-01-07"]), "site": ["Newport"],
                             "actual_da": [3.0], "predicted_da": [2.5], "actual_category": [0],
                             "predicted_category": [0]})


def test_enhanced_endpoint_serves_hits_and_computes_misses(store, monkeypatch):
    from backend import api

    engine = _Engine()
    monkeypatch.setattr(api, "get_forecast_engine", lambda: engine)
    body = {"date": "2019-06-03", "site": "Newport", "task": "regression", "model": "ensemble"}
    with TestClient(api.app) as client:
        served = client.post("/api/forecast/enhanced", json=body).json()
        assert engine.calls == [] and served["source"] == "precomputed"
        assert served["model_version"] == "datect-forecaster/7"

        monkeypatch.delenv(pf.STORE_ENV)
        computed = client.post("/api/forecast/enhanced", json=body).json()
        assert len(engine.calls) == 1 and computed["source"] == "computed"
        assert computed["model_version"] is None

        monkeypatch.setenv(pf.STORE_ENV, store)
        missed = client.post("/api/forecast/enhanced", json={**body, "model": "naive"}).json()
        assert missed["source"] == "computed" and len(engine.calls) == 2

    # Same forecast either way: the responses differ only in where they came from
    strip = ("source", "model_version")
    assert {k: v for k, v in served.items() if k not in strip} == \
        {k: v for k, v in computed.items() if k not in strip}


def test_visualizations_serve_the_store(store, monkeypatch):
    from backend import api

    calls = []
    monkeypatch.setattr(api, "generate_spectral_analysis",
                        lambda data, site, **kw: calls.append((site, kw)) or [{"computed": True}])
    with TestClient(api.app) as client:
        assert client.get("/api/visualizations/correlation/Newport").json()["source"] == "precomputed"
        assert client.get("/api/visualizations/spectral/all").json()["plots"][0]["layout"]["title"] == "all"
        assert calls == []
        computed = client.get("/api/visualizations/spectral/Coos Bay").json()
    # A miss computes, with the store's XGBoost and Random Forest slices as model traces
    assert computed["source"] == "computed" and calls[0][0] == "Coos Bay"
    pd.testing.assert_frame_equal(calls[0][1]["xgb_results"], pd.DataFrame(_records("xgboost")))
    pd.testing.assert_frame_equal(calls[0][1]["rf_results"], pd.DataFrame(_records("rf")))


def test_spectral_traces_from_records_match_a_results_frame():
    # The store keeps retrospective records (date strings); the file cache kept the
    # results DataFrame (datetimes). Both must give the same spectral plots.
    from backend.visualizations import generate_spectral_analysis

    data = pd.read_parquet(f"{pf.REPO_ROOT}/data/processed/final_output.parquet")
    records = pd.DataFrame(_records("xgboost") * 3)
    frame = records.assign(date=pd.to_datetime(records["date"]))
    a = generate_spectral_analysis(data.copy(), "Newport", records, records.assign(predicted_da=records["predicted_da"] + 1))
    b = generate_spectral_analysis(data.copy(), "Newport", frame, frame.assign(predicted_da=frame["predicted_da"] + 1))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert any(t["name"] == "XGBoost Predictions" for t in a[0]["data"])


def test_health_reports_the_store(store):
    from backend import api

    with TestClient(api.app) as client:
        health = client.get("/health").json()
    assert health["precomputed"]["tables"]["forecasts"]["entries"] == 3
    assert health["precomputed"]["tables"]["retrospective"]["entries"] == 2


def test_health_without_a_store(monkeypatch):
    from backend import api

    monkeypatch.delenv(pf.STORE_ENV, raising=False)
    with TestClient(api.app) as client:
        assert client.get("/health").json()["precomputed"] is None


def test_unreadable_store_falls_back_without_retrying(tmp_path, monkeypatch):
    bad = tmp_path / "forecasts"
    bad.mkdir()
    (bad / "_delta_log").mkdir()
    (bad / "_delta_log" / "00000000000000000000.json").write_text("not json")
    monkeypatch.setenv(pf.STORE_ENV, str(tmp_path))
    _reset(monkeypatch)
    assert pf.forecast("Newport", "2019-06-03", "regression", "ensemble") is None
    assert pf._loaded_root == str(tmp_path)  # remembered: no reload per request
    assert "error" in pf.status()["tables"]["forecasts"]


def test_check_requires_every_table_current(store, tmp_path):
    # The image build runs this: a deployed store must be complete and current,
    # or the API would recompute retrospectives (minutes each) on Cloud Run
    problems = pf.check_store(store)
    assert any("forecasts" in p and "stale" in p for p in problems)  # the fixture's stale row
    assert pf.check_store(str(tmp_path / "nowhere")) == [f"{tmp_path / 'nowhere'}: no such directory"]
    deltalake = pytest.importorskip("deltalake")
    good = tmp_path / "good"
    current = pf.config_hash()
    for name, row in {"forecasts": {"site": "Newport", "forecast_date": datetime.date(2019, 6, 3),
                                    "task": "regression", "model_type": "ensemble",
                                    "result_json": pf.encode(None), "model_version": "v"},
                      "retrospective": {"task": "regression", "model_type": "ensemble", "records_json": "[]"},
                      "spectral": {"site": pf.ALL_SITES, "plots_json": "[]"},
                      "correlation": {"site": "Newport", "plot_json": "{}"}}.items():
        deltalake.write_deltalake(str(good / name), pd.DataFrame([{**row, "config_hash": current}]))
    assert pf.check_store(str(good)) == []
    assert pf.check_store(str(good), require=("forecasts", "nope")) == ["nope: missing"]


def test_one_bad_table_does_not_disable_the_others(store, monkeypatch):
    import shutil

    shutil.rmtree(f"{store}/spectral/_delta_log")
    import os
    os.makedirs(f"{store}/spectral/_delta_log")
    with open(f"{store}/spectral/_delta_log/00000000000000000000.json", "w") as f:
        f.write("not json")
    _reset(monkeypatch)
    assert pf.correlation("Newport") == {"data": [{"z": [[1.0]]}]}
    assert pf.spectral(None) is None
    assert "error" in pf.status()["tables"]["spectral"]


def test_config_hash_covers_row_serialization_and_python(monkeypatch, tmp_path):
    import shutil

    root = tmp_path / "repo"
    for rel in ("forecasting", "config", "data/processed", "data/raw/da-input", "backend"):
        (root / rel).mkdir(parents=True)
    for f in pf.glob.glob(f"{pf.REPO_ROOT}/forecasting/*.py"):
        shutil.copy(f, root / "forecasting")
    for rel in ("config.py", "config/tuned_hyperparameters.json", "data/processed/final_output.parquet",
                "backend/visualizations.py", "backend/records.py"):
        shutil.copy(f"{pf.REPO_ROOT}/{rel}", root / rel)
    first = pf.config_hash(str(root))
    (root / "backend" / "records.py").write_text("# a different record format\n")
    assert pf.config_hash(str(root)) != first
    before = pf.config_hash()
    monkeypatch.setattr(pf.platform, "python_version_tuple", lambda: ("3", "99", "0"))
    assert pf.config_hash() != before


def test_status_says_when_the_directory_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv(pf.STORE_ENV, str(tmp_path / "typo"))
    _reset(monkeypatch)
    assert pf.status()["exists"] is False


def test_write_sample_stores_regression_forecasts_only(tmp_path, monkeypatch):
    deltalake = pytest.importorskip("deltalake")
    import forecasting

    class _Stub:
        def __init__(self, **kw):
            pass

        def generate_single_forecast(self, data_path, forecast_date, site, task, model_type):
            return _result(site, pd.Timestamp(forecast_date).date().isoformat(), task)

    monkeypatch.setattr(forecasting, "ForecastEngine", _Stub)
    assert pf.write_sample(str(tmp_path), [("Newport", "2019-06-03"), ("Kalaloch", "2019-06-03")]) == 2
    tasks = deltalake.DeltaTable(str(tmp_path / "forecasts")).to_pandas()["task"]
    assert set(tasks) == {"regression"}


def test_store_builds_only_the_slices_spectral_reads():
    # Read the assignment from source: importing datect_spark may need Spark
    import ast

    tree = ast.parse(open(f"{pf.REPO_ROOT}/datect_spark/precompute_store.py").read())
    value = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", None) == "SLICES" for t in n.targets))
    assert ast.literal_eval(value) == (("regression", "xgboost"), ("regression", "rf"))

"""The Spark weight tuner's contract with the engine (datect_spark/retune.py). Its
partition function runs without Spark, so it is checked here on real tasks."""
import math
import os
import time

import pandas as pd
import pytest

from forecasting.raw_data_forecaster import load_raw_da_measurements


@pytest.fixture(scope="module")
def retune():
    # datect_spark.session pins TZ=UTC at import; give the rest of the suite its zone back
    saved = os.environ.get("TZ")
    from datect_spark import retune
    if saved is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = saved
    time.tzset()
    return retune


# Two real measurements, and a site's first measurement (no training data: ok=False)
TASKS = (("Newport", "2019-06-06"), ("Twin Harbors", "2022-09-06"), ("Long Beach", "2002-01-14"))


def _tasks() -> pd.DataFrame:
    """The input predict() hands each partition: site, YYYY-MM-DD date, actual_da."""
    raw = load_raw_da_measurements()
    raw["date"] = pd.to_datetime(raw["date"])
    rows = []
    for site, day in TASKS:
        match = raw[(raw["site"] == site) & (raw["date"] == pd.Timestamp(day))]
        assert len(match) == 1, (site, day)
        rows.append((site, day, float(match["da_raw"].iloc[0])))
    return pd.DataFrame(rows, columns=["site", "date", "actual_da"])


@pytest.mark.usefixtures("fast_models")
class TestRetuneContract:
    def test_partition_reports_the_engines_point_forecast(self, retune, monkeypatch):
        import config

        # _engine() builds the engine with fast_models' params and chdirs to the repo
        # root: both are undone after the test
        monkeypatch.setattr(retune, "_ENGINE", None)
        monkeypatch.chdir(retune.REPO_ROOT)

        frames = list(retune._predict_partition(iter([_tasks()])))
        assert len(frames) == 1
        out = frames[0]
        assert list(out.columns) == ["site", "date", "actual_da", "xgb_raw", "rf_raw",
                                     "n_train", "task_seconds", "host", "ok"]
        assert list(out.columns) == [c.split()[0] for c in retune.PRED_SCHEMA.split(", ")]
        assert [(r.site, r.date) for r in out.itertuples()] == list(TASKS)

        engine, feature_frame, base = retune._ENGINE
        assert base == {**config.XGB_REGRESSION_PARAMS, "n_jobs": 1}
        for r in out.itertuples():
            core = engine._point_forecast(
                {"date": pd.Timestamp(r.date), "site": r.site, "da_raw": r.actual_da},
                feature_frame, {**config.XGB_REGRESSION_PARAMS, "n_jobs": 1},
            )
            if core is None:
                assert not r.ok and r.n_train == 0
                assert math.isnan(r.xgb_raw) and math.isnan(r.rf_raw)
                continue
            assert r.ok
            assert r.n_train == core["training_samples"] > 0
            assert r.xgb_raw == core["predicted_da"]
            assert r.rf_raw == core["predicted_da_rf"]
        assert out["ok"].tolist() == [True, True, False]

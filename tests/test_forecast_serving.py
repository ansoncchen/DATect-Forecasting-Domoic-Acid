"""Spark precompute of dashboard forecasts (datect_spark/forecast_serving.py): the
checks that run without Spark."""
import os
import time

import pytest


@pytest.fixture(scope="module")
def serving():
    # datect_spark.session pins TZ=UTC at import; give the rest of the suite its zone back
    saved = os.environ.get("TZ")
    from datect_spark import forecast_serving
    if saved is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = saved
    time.tzset()
    return forecast_serving


def test_forecasts_table_keeps_task_and_stores_no_interval(serving):
    # The engine computes no interval any more; the table keeps its task column
    columns = [c.split()[0] for c in serving.OUT_SCHEMA.split(", ")]
    assert columns == ["site", "forecast_date", "task", "model_type", "predicted_da",
                       "predicted_category", "spike_probability", "spike_alert",
                       "training_samples", "result_json", "error", "task_seconds", "host"]
    summary = serving._summary({"predicted_da": 5.3, "predicted_category": 1,
                                "spike_probability": 0.02, "spike_alert": False,
                                "training_samples": 600})
    assert set(summary) <= set(columns)


def test_precompute_tasks_are_regression_only(serving):
    # The dashboard's forecast is one regression run; categories come from the band
    from backend import precomputed
    assert precomputed.FORECAST_TASKS == ("regression",)
    assert serving.TASKS is precomputed.FORECAST_TASKS

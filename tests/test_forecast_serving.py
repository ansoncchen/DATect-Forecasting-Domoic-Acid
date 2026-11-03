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


def test_bootstrap_interval_is_not_stored(serving):
    # The bootstrap fallback draws from the global NumPy RNG: not reproducible on demand
    fell_back = {"bootstrap_quantiles": {"q05": 1.0, "q50": 2.0, "q95": 3.0,
                                         "bootstrap_predictions": [1.0, 2.0]}}
    assert "bootstrap" in serving.not_reproducible(fell_back)
    assert serving.not_reproducible({"bootstrap_quantiles": {"q05": 1.0, "q50": 2.0, "q95": 3.0}}) is None
    assert serving.not_reproducible(None) is None


def test_classification_model_matches_the_api(serving):
    # api.py asks the engine for "logistic" when the dashboard's model is "linear"
    assert serving.task_model("classification", "linear") == "logistic"
    assert serving.task_model("regression", "linear") == "linear"
    assert serving.task_model("classification", "ensemble") == "ensemble"

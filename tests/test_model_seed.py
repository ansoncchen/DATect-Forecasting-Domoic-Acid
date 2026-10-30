"""Models are always seeded with config.MODEL_SEED, never the sampling seed.

Evaluation scripts change config.RANDOM_SEED to draw different test points.
Joblib/Spark workers re-import config, so if models used RANDOM_SEED a serial
run and a parallel run of the same evaluation would train different models.
"""
import config
from forecasting.raw_model_factory import (
    build_rf_regressor,
    build_xgb_classifier,
    build_xgb_regressor,
)


def test_models_ignore_the_sampling_seed(monkeypatch):
    monkeypatch.setattr(config, "RANDOM_SEED", 123)
    assert build_xgb_regressor({}).random_state == config.MODEL_SEED
    assert build_xgb_classifier({}).random_state == config.MODEL_SEED
    assert build_rf_regressor({}).random_state == config.MODEL_SEED

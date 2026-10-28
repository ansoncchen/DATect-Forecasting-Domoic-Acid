"""Shared pytest fixtures."""
import pytest


@pytest.fixture(scope="class")
def fast_models(request):
    """Shrink every model so integration tests run in seconds, then restore config.

    Model parameter dicts are patched in place with ``setitem`` because several
    modules hold direct references to them (and per-site dicts are shared with
    the cached tuned-hyperparameter JSON); ``undo()`` restores every entry after
    the class. A test class may set ``test_sample_fraction`` to control
    retrospective size.
    """
    import config
    import forecasting.per_site_models as per_site_models

    mp = pytest.MonkeyPatch()
    mp.setattr(config, "N_BOOTSTRAP_ITERATIONS", 2)
    for name in ("XGB_REGRESSION_PARAMS", "RF_REGRESSION_PARAMS",
                 "XGB_CLASSIFICATION_PARAMS", "SPIKE_CLASSIFIER_PARAMS"):
        mp.setitem(getattr(config, name), "n_estimators", 5)
    mp.setattr(config, "ENABLE_PARALLEL", False)
    mp.setattr(config, "MIN_TRAINING_FOR_TUNING", 99999)
    fraction = getattr(request.cls, "test_sample_fraction", None)
    if fraction is not None:
        mp.setattr(config, "TEST_SAMPLE_FRACTION", fraction)

    for site_cfg in per_site_models.SITE_SPECIFIC_CONFIGS.values():
        for key in ("xgb_params", "rf_params"):
            if site_cfg.get(key):
                mp.setitem(site_cfg[key], "n_estimators", 5)

    yield
    mp.undo()


@pytest.fixture
def fresh_import(monkeypatch):
    """Drop matching modules so a test re-imports them; restore the originals afterwards.

    Without the restore, later tests (and the already-imported engine) would see two
    copies of config / per_site_models / tuned_config and patch the wrong one.
    Usage: ``fresh_import(lambda name: "tuned_config" in name)``.
    """
    import sys

    import forecasting

    def _drop(match):
        for name in [m for m in list(sys.modules) if match(m)]:
            parent, _, child = name.rpartition(".")
            if parent == "forecasting" and hasattr(forecasting, child):
                # re-importing rebinds the package attribute; register the original
                monkeypatch.setattr(forecasting, child, getattr(forecasting, child))
            monkeypatch.delitem(sys.modules, name)

    return _drop

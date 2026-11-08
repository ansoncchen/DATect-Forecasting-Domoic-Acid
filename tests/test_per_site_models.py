"""The one ensemble blend and its default weights."""

import numpy as np

from forecasting.per_site_models import SITE_SPECIFIC_CONFIGS


def test_blend_matches_the_engines_formulas_bitwise():
    from forecasting.per_site_models import blend, get_site_ensemble_weights

    rng = np.random.default_rng(0)
    for site in SITE_SPECIFIC_CONFIGS:
        w_xgb, w_rf, w_naive = get_site_ensemble_weights(site)
        for _ in range(100):
            x, r, n = (float(v) for v in rng.uniform(-5, 80, 3))
            # engine: single forecast / retrospective (with naive term)
            assert blend(x, r, n, (w_xgb, w_rf, w_naive)) == (
                w_xgb * x + w_rf * r + w_naive * n
            )
            # engine: spike alert (no naive term)
            assert blend(x, r, 0.0, (w_xgb, w_rf, 0.0)) == w_xgb * x + w_rf * r
            # missing RF falls back to the XGB prediction
            assert blend(x, None, n, (w_xgb, w_rf, w_naive)) == (
                w_xgb * x + w_rf * x + w_naive * n
            )


def test_blend_is_elementwise_on_arrays():
    from forecasting.per_site_models import blend

    rng = np.random.default_rng(1)
    x, r, n = rng.uniform(0, 50, (3, 200))
    w = (0.5, 0.5, 0.0)
    assert (blend(x, r, n, w) == w[0] * x + w[1] * r + w[2] * n).all()


def test_equal_weights_when_per_site_models_are_off(monkeypatch, fresh_import):
    monkeypatch.setenv("DATECT_USE_PER_SITE_MODELS", "false")
    fresh_import(lambda m: m == "config" or m == "forecasting.per_site_models")
    from forecasting.per_site_models import get_site_ensemble_weights

    assert get_site_ensemble_weights("Newport") == (0.5, 0.5, 0.0)

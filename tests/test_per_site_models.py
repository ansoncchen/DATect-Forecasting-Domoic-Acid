"""The one blend and its default weights (the random forest alone)."""

import numpy as np

from forecasting.per_site_models import SITE_SPECIFIC_CONFIGS


def _bits(values) -> np.ndarray:
    """The IEEE-754 bit patterns, so 'equal' means bit for bit (NaNs and signed zeros too)."""
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def test_blend_matches_the_engines_formulas_bitwise():
    """blend is w_xgb*x + w_rf*r + w_naive*n evaluated left to right, as the inline
    formulas it replaced. Every tuned naive weight is 0, which hides the evaluation
    order, so a (0.3, 0.3, 0.4) draw is added."""
    from forecasting.per_site_models import blend

    weight_sets = [SITE_SPECIFIC_CONFIGS[site]["ensemble_weights"] for site in SITE_SPECIFIC_CONFIGS]
    rng = np.random.default_rng(0)
    for weights in weight_sets + [(0.3, 0.3, 0.4)]:
        w_xgb, w_rf, w_naive = weights
        order_shows = False
        for _ in range(100):
            x, r, n = (float(v) for v in rng.uniform(-5, 80, 3))
            # engine: single forecast / retrospective (with naive term)
            assert _bits(blend(x, r, n, weights)) == _bits(w_xgb * x + w_rf * r + w_naive * n)
            order_shows |= bool(_bits((w_xgb * x + w_rf * r) + w_naive * n)
                                != _bits(w_xgb * x + (w_rf * r + w_naive * n)))
            # engine: spike alert (no naive term)
            assert _bits(blend(x, r, 0.0, (w_xgb, w_rf, 0.0))) == _bits(w_xgb * x + w_rf * r)
            # missing RF falls back to the XGB prediction
            assert _bits(blend(x, None, n, weights)) == _bits(w_xgb * x + w_rf * x + w_naive * n)
        if w_naive:
            assert order_shows, "this draw cannot tell the evaluation order apart"


def test_blend_is_elementwise_on_arrays():
    from forecasting.per_site_models import blend

    rng = np.random.default_rng(1)
    x, r, n = rng.uniform(0, 50, (3, 200))
    w = (0.5, 0.5, 0.0)
    assert (blend(x, r, n, w) == w[0] * x + w[1] * r + w[2] * n).all()


def test_the_random_forest_alone_is_the_default(monkeypatch, fresh_import):
    """One configuration at every beach (docs/FORECAST_EXPERIMENTS.md §3c); the retired
    per-site weights come back only when switched on."""
    monkeypatch.delenv("DATECT_USE_PER_SITE_MODELS", raising=False)
    fresh_import(lambda m: m == "config" or m == "forecasting.per_site_models")
    import config
    from forecasting.per_site_models import get_site_ensemble_weights

    assert config.USE_PER_SITE_MODELS is False
    assert {get_site_ensemble_weights(s) for s in SITE_SPECIFIC_CONFIGS} == {(0.0, 1.0, 0.0)}


def test_per_site_weights_only_when_switched_on(monkeypatch, fresh_import):
    monkeypatch.setenv("DATECT_USE_PER_SITE_MODELS", "true")
    fresh_import(lambda m: m == "config" or m == "forecasting.per_site_models")
    from forecasting.per_site_models import get_site_ensemble_weights

    assert get_site_ensemble_weights("Newport") == SITE_SPECIFIC_CONFIGS["Newport"]["ensemble_weights"]


def test_unknown_site_gets_the_shared_weights(monkeypatch):
    import forecasting.per_site_models as per_site_models

    monkeypatch.setattr(per_site_models.config, "USE_PER_SITE_MODELS", True)
    assert "Nowhere Beach" not in SITE_SPECIFIC_CONFIGS
    assert per_site_models.get_site_ensemble_weights("Nowhere Beach") == per_site_models.SHARED_WEIGHTS
    assert per_site_models.SHARED_WEIGHTS == (0.0, 1.0, 0.0)


def test_shared_blend_is_the_random_forest(monkeypatch):
    """With per-site models off, the retrospective's per-row blend (iterrows over the
    results frame) is the random forest's prediction, including a row whose RF prediction
    is NaN."""
    import pandas as pd

    import forecasting.per_site_models as per_site_models

    monkeypatch.setattr(per_site_models.config, "USE_PER_SITE_MODELS", False)
    rng = np.random.default_rng(2)
    sites = list(SITE_SPECIFIC_CONFIGS)
    # What the engine's per-point results hold: a string site, Python-float predictions
    results = [{"site": sites[i % len(sites)], "predicted_da": float(x),
                "predicted_da_rf": float(r), "naive_prediction": float(n)}
               for i, (x, r, n) in enumerate(rng.uniform(0, 80, (500, 3)))]
    results[7]["predicted_da_rf"] = float("nan")
    df = pd.DataFrame(results)

    blended = np.array([per_site_models.blend(row["predicted_da"], row["predicted_da_rf"], row["naive_prediction"],
                                              per_site_models.get_site_ensemble_weights(row["site"]))
                        for _, row in df.iterrows()])
    assert np.isnan(blended[7])
    assert np.array_equal(blended, df["predicted_da_rf"].to_numpy(), equal_nan=True)

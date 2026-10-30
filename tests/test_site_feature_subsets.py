"""Every per-site feature_subset must name columns that exist in the feature frame.

compute_site_drop_cols() keeps only subset columns, so a misspelled or removed
feature would otherwise be dropped from training without any error.
"""
import pytest

from forecasting.feature_utils import add_temporal_features
from forecasting.per_site_models import SITE_SPECIFIC_CONFIGS
from forecasting.raw_forecast_engine import RawForecastEngine


@pytest.fixture(scope="module")
def feature_columns():
    engine = RawForecastEngine(validate_on_init=False)
    return set(add_temporal_features(engine._load_feature_frame()).columns)


@pytest.mark.parametrize("site", sorted(SITE_SPECIFIC_CONFIGS))
def test_feature_subset_columns_exist(site, feature_columns):
    subset = SITE_SPECIFIC_CONFIGS[site].get("feature_subset") or []
    missing = sorted(set(subset) - feature_columns)
    assert not missing, f"{site} feature_subset lists unknown columns: {missing}"

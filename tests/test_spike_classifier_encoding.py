"""The pooled spike classifier must score test rows on its own feature encoding.

It is trained on all sites' rows encoded by a pooled imputer/MinMax transformer.
Scoring a test row encoded by the per-site transformer (different scaling, site
feature subset, zero-filled gaps) silently skews the spike probability.
"""
import pandas as pd

from forecasting.feature_utils import add_temporal_features
from forecasting.raw_forecast_engine import RawForecastEngine


def test_test_row_is_encoded_with_the_pooled_transformer():
    engine = RawForecastEngine(validate_on_init=False)
    frame = engine._load_feature_frame()
    anchor = pd.Timestamp("2020-06-01")

    X_pooled, _, transformer, columns = engine._build_pooled_spike_data(frame, anchor)
    row = add_temporal_features(frame[(frame["site"] == "Copalis") & (frame["date"] <= anchor)].tail(1))
    encoded = transformer.transform(row.reindex(columns=columns))

    assert list(encoded.columns) == list(X_pooled.columns)
    assert encoded.notna().all().all()  # gaps get the pooled median, not NaN or 0-by-default

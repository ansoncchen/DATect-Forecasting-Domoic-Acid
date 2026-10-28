"""Raw measurement parsing rules (forecasting/measurements.py)."""
import math

import numpy as np
import pandas as pd

from forecasting.measurements import parse_da_values, parse_dates, parse_pn_values


def test_below_detection_da_is_half_the_limit():
    out = parse_da_values(pd.Series(["1.2", "<1", " < 2 ", "abc", None, 3.5], dtype=object))
    assert out.tolist()[:3] == [1.2, 0.5, 1.0]
    assert math.isnan(out[3]) and math.isnan(out[4])
    assert out[5] == 3.5


def test_numeric_da_column_passes_through():
    assert parse_da_values(pd.Series([1.0, 2.5])).tolist() == [1.0, 2.5]


def test_pn_thousands_separators():
    out = parse_pn_values(pd.Series(["12,345", "1,234,567", "500", "1,234.5"], dtype=object))
    assert out.tolist() == [12345.0, 1234567.0, 500.0, 1234.5]


def test_pn_ambiguous_grouping_is_not_guessed():
    # "999,99" is neither a thousands-grouped integer nor a dot decimal
    assert np.isnan(parse_pn_values(pd.Series(["999,99"], dtype=object))[0])


def test_dates_tolerate_whitespace():
    out = parse_dates(pd.Series([" 7/11/19", "7/12/19"], dtype=object))
    assert out.tolist() == [pd.Timestamp("2019-07-11"), pd.Timestamp("2019-07-12")]

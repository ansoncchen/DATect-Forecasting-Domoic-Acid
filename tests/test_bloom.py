"""The bloom Watch (forecasting/bloom.py): a Pseudo-nitzschia bloom at a Washington
beach while its last toxin reading is still low. The rule was declared and backtested
in the onset research (docs/ONSET_RESEARCH.md, sections 13-15, on the research
branch); the expected flags below are that research's port spec."""
import numpy as np
import pandas as pd
import pytest

from forecasting import bloom

ALL_YEARS = "results/chronological/chronological_regression_ensemble_20030101_20240101.parquet"


def counts(site, samples):
    """A count table for one site from (ISO date, cells/L) pairs."""
    dates = np.array([np.datetime64(d) for d, _ in samples], dtype="datetime64[D]")
    return {site: (dates, np.array([c for _, c in samples], dtype=float))}


def on(site, anchor, last, table):
    return bloom.bloom_watch(site, pd.Timestamp(anchor), last, table)


def test_window_is_the_seven_days_ending_on_the_anchor():
    c = counts("Long Beach", [("2020-10-08", 100_000.0)])
    assert on("Long Beach", "2020-10-08", 1.0, c)          # the anchor day counts
    assert on("Long Beach", "2020-10-14", 1.0, c)          # six days later: still inside
    assert not on("Long Beach", "2020-10-15", 1.0, c)      # anchor - 7 is outside
    assert not on("Long Beach", "2020-10-07", 1.0, c)      # a count after the anchor is the future


def test_threshold_and_last_value_boundaries():
    at = counts("Long Beach", [("2020-10-08", 100_000.0)])
    under = counts("Long Beach", [("2020-10-08", 99_999.0)])
    assert on("Long Beach", "2020-10-08", 4.99, at)
    assert not on("Long Beach", "2020-10-08", 1.0, under)  # a count at the threshold counts, below does not
    assert not on("Long Beach", "2020-10-08", 5.0, at)     # last value must be below 5
    assert not on("Long Beach", "2020-10-08", float("nan"), at)
    assert not on("Long Beach", "2020-10-08", None, at)


def test_washington_only_and_the_beach_own_counts():
    assert not on("Newport", "2020-10-08", 1.0, counts("Newport", [("2020-10-08", 1e6)]))
    assert not on("Long Beach", "2020-10-08", 1.0, counts("Twin Harbors", [("2020-10-08", 1e6)]))
    assert not on("Long Beach", "2020-10-08", 1.0, counts("Long Beach", []))


def test_loader_reads_every_washington_beach():
    """Sample-days after collapsing same-day samples to their maximum (port spec, section 2)."""
    table = bloom.load_pn_counts()
    expected = {
        "Kalaloch": (1537, "2002-01-10", "2023-11-02"),
        "Quinault": (1367, "2000-06-21", "2023-12-19"),
        "Copalis": (1587, "2002-01-10", "2023-12-19"),
        "Twin Harbors": (1962, "2002-01-02", "2023-12-29"),
        "Long Beach": (1983, "2002-01-02", "2023-12-27"),
    }
    for site, (n, first, last) in expected.items():
        dates, values = table[site]
        assert (len(dates), str(dates[0]), str(dates[-1])) == (n, first, last), site
        assert np.all(np.diff(dates.astype("int64")) > 0) and np.isfinite(values).all()


@pytest.fixture(scope="module")
def all_years():
    rows = pd.read_parquet(ALL_YEARS)
    rows["date"] = pd.to_datetime(rows["date"])
    rows["anchor_date"] = pd.to_datetime(rows["anchor_date"])
    rows["bloom"] = [bloom.bloom_watch(r.site, r.anchor_date, r.naive_prediction) for r in rows.itertuples()]
    return rows


def test_flags_on_the_all_years_run_match_the_research(all_years):
    flagged = all_years[all_years["bloom"]]
    assert len(flagged) == 92
    year = flagged["date"].dt.year
    assert [int((year < 2015).sum()), int(year.between(2015, 2018).sum()),
            int(year.between(2019, 2021).sum()), int((year >= 2022).sum())] == [52, 12, 19, 9]
    assert set(flagged["site"]) <= {"Kalaloch", "Quinault", "Copalis", "Twin Harbors", "Long Beach"}
    assert (flagged["naive_prediction"] < 5).all()
    # the one sudden rise it adds in the validation window
    assert ((flagged["site"] == "Long Beach") & (flagged["date"] == "2020-10-22")).any()

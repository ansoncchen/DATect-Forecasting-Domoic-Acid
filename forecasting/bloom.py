"""
Bloom Watch: a Pseudo-nitzschia bloom at a Washington beach while its toxin is still low.

The forecast and the classifier miss sudden rises from near zero: in 2015-2021 the hybrid
alert caught none of Washington's 8 rises from below 5 to 20 ug/g or more. The onset
research found one signal that precedes them, the beach's own cell count, and declared and
backtested this rule before it was ported (docs/ONSET_RESEARCH.md, sections 13-15, on the
research branch): leave-one-event-out 10 of 14 at a 10% budget, and a permutation test
that re-runs the whole search gives p = 0.012 at that budget. Seven bloom events stand
behind it, so its record is rough.

Leak-free: only samples dated on or before the anchor are read, like the DA readings the
engine sees. A count is taken as known on its sample day.

The counts are ORHAB data (Olympic Region Harmful Algal Blooms partnership). The rule is a
serve-time alert reason: it changes no prediction, so the engine's ``spike_alert`` stays
the hybrid alert and this module only adds ``by_bloom`` to ``metrics.alert_parts``.
"""
from __future__ import annotations

import os
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

import config

from .measurements import parse_dates, parse_pn_values

Counts = dict[str, tuple[np.ndarray, np.ndarray]]


def _site_name(file_key: str) -> str:
    return file_key.replace("-pn", "").replace("-", " ").title()


@lru_cache(maxsize=1)
def load_pn_counts() -> Counts:
    """Raw Pseudo-nitzschia counts per site: (sorted sample days, the day's highest count).

    Read from ``config.ORIGINAL_PN_FILES`` with the shared parsers; same-day samples
    collapse to their maximum and nothing is gap-filled. The weekly ``pn`` column of
    final_output.parquet is not used: it is aggregated by week and decay-filled.
    """
    out: Counts = {}
    for key, path in config.ORIGINAL_PN_FILES.items():
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        value_cols = [c for c in frame.columns if "pseudo" in str(c).lower()]
        if "Date" not in frame.columns or len(value_cols) != 1:
            continue
        with warnings.catch_warnings():  # the Washington files mix 1/2/2002 and 01/10/02
            warnings.simplefilter("ignore")
            dates = parse_dates(frame["Date"])
        day = pd.DataFrame({"date": dates, "pn": parse_pn_values(frame[value_cols[0]])}).dropna()
        day = day.groupby("date", as_index=False)["pn"].max().sort_values("date")
        out[_site_name(key)] = (day["date"].to_numpy("datetime64[D]"), day["pn"].to_numpy(float))
    return out


def applies(site: str) -> bool:
    """Whether the bloom Watch covers ``site`` (Washington beaches)."""
    return config.SITE_STATES.get(site) in config.BLOOM_WATCH_STATES


def _window(site: str, anchor, table: Counts) -> tuple[np.ndarray, np.ndarray]:
    dates, values = table.get(site, (np.array([], dtype="datetime64[D]"), np.array([])))
    a = np.datetime64(pd.Timestamp(anchor).date())
    lo = int(np.searchsorted(dates, a - np.timedelta64(config.BLOOM_WATCH_DAYS, "D"), side="right"))
    hi = int(np.searchsorted(dates, a, side="right"))
    return dates[lo:hi], values[lo:hi]


def bloom_watch(site: str, anchor, last_known: float | None, table: Counts | None = None) -> bool:
    """True when ``site`` is covered, its last DA reading is below the cap, and its own
    count reached the threshold on a day in (anchor - BLOOM_WATCH_DAYS, anchor]."""
    if not applies(site) or last_known is None or not last_known < config.BLOOM_WATCH_LAST_BELOW:
        return False  # NaN fails the comparison
    _, values = _window(site, anchor, load_pn_counts() if table is None else table)
    return values.size > 0 and float(values.max()) >= config.BLOOM_WATCH_COUNT


def bloom_status(site: str, anchor, table: Counts | None = None) -> dict:
    """What the dashboard says about the bloom Watch for one forecast: whether it covers the
    site, whether any count fell in the window, and the newest count on or before the anchor."""
    if not applies(site):
        return {"applies": False, "counted": False, "last_count_date": None}
    table = load_pn_counts() if table is None else table
    in_window, _ = _window(site, anchor, table)
    dates = table.get(site, (np.array([], dtype="datetime64[D]"),))[0]
    upto = dates[dates <= np.datetime64(pd.Timestamp(anchor).date())]
    return {"applies": True, "counted": bool(in_window.size),
            "last_count_date": str(upto[-1]) if upto.size else None}

"""Transport-trigger features (docs/FORECAST_EXPERIMENTS.md §7): NDBC 46041 wind and the
NEMO/ChaBa mooring, as of an information cutoff (days <= cutoff, UTC daily aggregates).

The inputs are not in the repo. ``DATECT_TT_WIND`` points at the hourly NDBC 46041
parquet the old wind chain cached (``git show e4a15a8c^:data/raw/ndbc_46041_wind.parquet``);
``DATECT_TT_NEMO`` at PMEL's ``pCO2105_ALLdata_nemo23a.txt``. The mooring data are
unverified and shared for partners' qualitative use: nothing derived from them per
timestamp may be written into the repo.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

WIND_COLS = ["wind_v_1d", "wind_v_3d", "wind_v_7d", "wind_upw_14d", "wind_reversal_7d",
             "wind_days_since_reversal", "wind_coverage"]
NEMO_COLS = ["nemo_sst_rise", "nemo_sst_trend_7d", "nemo_o2_min_7d", "nemo_ph_min_7d",
             "nemo_chl_3d", "nemo_chl_max_7d", "nemo_coverage"]
INTERACTION_COLS = ["rev_x_pn", "rev_x_nb_max", "rev_x_last_da"]
# §7.1 amendment: the beach PN bloom's shape (H-crash) and the offshore OAD score 2-8 weeks back (H-sat-lag)
PN_CRASH_COLS = ["pn_peak_8w_log", "pn_drop_frac", "pn_weeks_since_peak", "pn_crash"]
SAT_COLS = ["sat_max_2_8w", "sat_decline", "sat_valid_2_8w"]
PN_BLOOM, PN_CRASH_DROP, PN_WINDOW = 50_000.0, 0.9, 56
SAT_SHIFT, SAT_SPAN, SAT_MIN_VALID = 5, (14, 56), 0.2
SAT_SOURCE = {s: "Olympic Coast (WA)" for s in
              ("Kalaloch", "Quinault", "Copalis", "Twin Harbors", "Long Beach", "Clatsop Beach", "Cannon Beach")}
SAT_SOURCE.update({"Newport": "Central Oregon", "Coos Bay": "Central Oregon", "Gold Beach": "Southern OR / N CA"})
REVERSAL_CAP = 60
NEMO_RANGES = {"sst": (5, 25), "sss": (25, 35), "chl": (0, 50), "o2": (0, 20), "ph": (7.5, 8.7)}
NEMO_SOURCE = {"Sea Surface Temperature (deg C)": "sst", "Sea Surface Salinity": "sss",
               "Chlorophyll (ug/l)": "chl", "Salinity-Compensated O2 (mg/l)": "o2",
               "pH of Seawater (total scale)": "ph"}
DAY0 = np.datetime64("2000-01-01", "D")

_DAILY: dict = {}
_CACHE: dict = {}


def _grid(series: pd.Series) -> np.ndarray:
    """Daily values on a fixed grid from DAY0 (index = days since DAY0), NaN where missing."""
    idx = ((series.index.values.astype("datetime64[D]") - DAY0) / np.timedelta64(1, "D")).astype(int)
    out = np.full(idx.max() + 400, np.nan)
    out[idx] = series.to_numpy(float)
    return out


def wind_daily() -> dict:
    if "wind" not in _DAILY:
        w = pd.read_parquet(os.environ["DATECT_TT_WIND"]).dropna(subset=["wdir", "wspd"])
        v = -w["wspd"] * np.cos(np.radians(w["wdir"]))
        day = pd.DatetimeIndex(w["datetime"]).normalize()
        daily = v.groupby(day).mean()
        g = _grid(daily)
        r3 = pd.Series(g).rolling(3, min_periods=2).mean().to_numpy()
        cross = np.zeros(len(g), bool)
        cross[1:] = (r3[:-1] < 0) & (r3[1:] >= 0)
        _DAILY["wind"] = {"v": g, "cross": cross}
    return _DAILY["wind"]


def nemo_daily() -> dict:
    if "nemo" not in _DAILY:
        d = pd.read_csv(os.environ["DATECT_TT_NEMO"], skiprows=9, dtype=str)
        d.columns = [c.strip() for c in d.columns]
        day = pd.to_datetime(d["Date Time (UTC)"].str.strip(), errors="coerce").dt.normalize()
        out = {}
        for col, key in NEMO_SOURCE.items():
            x = pd.to_numeric(d[col].str.strip(), errors="coerce")
            lo, hi = NEMO_RANGES[key]
            x = x.where((x >= lo) & (x <= hi))
            s = x.groupby(day).mean().dropna()
            out[key] = _grid(s)
        _DAILY["nemo"] = out
    return _DAILY["nemo"]


def _day(ts) -> int:
    return int((np.datetime64(pd.Timestamp(ts), "D") - DAY0) / np.timedelta64(1, "D"))


def _window(arr: np.ndarray, c: int, n: int) -> np.ndarray:
    """The n daily values ending at day c (inclusive); NaN outside the grid."""
    lo = c - n + 1
    if c < 0 or lo >= len(arr):
        return np.full(n, np.nan)
    seg = arr[max(lo, 0): min(c + 1, len(arr))]
    pad_l, pad_r = max(0, -lo), max(0, c + 1 - len(arr))
    return np.concatenate([np.full(pad_l, np.nan), seg, np.full(pad_r, np.nan)])


def _mean(x: np.ndarray, need: float = 0.5) -> float:
    ok = ~np.isnan(x)
    return float(x[ok].mean()) if ok.mean() >= need and ok.any() else np.nan


def _stat(x: np.ndarray, fn, need: float = 0.5) -> float:
    ok = ~np.isnan(x)
    return float(fn(x[ok])) if ok.mean() >= need and ok.any() else np.nan


def wind_features(cutoff) -> dict:
    key = ("wind", _day(cutoff))
    if key in _CACHE:
        return _CACHE[key]
    w, c = wind_daily(), key[1]
    v = w["v"]
    f = {"wind_v_1d": _mean(_window(v, c, 1)), "wind_v_3d": _mean(_window(v, c, 3)),
         "wind_v_7d": _mean(_window(v, c, 7))}
    v14 = _window(v, c, 14)
    ok = ~np.isnan(v14)
    f["wind_coverage"] = float(ok.mean())
    f["wind_upw_14d"] = float(np.clip(-v14[ok], 0, None).sum()) if ok.mean() >= 0.5 else np.nan
    cross = w["cross"][max(0, c - REVERSAL_CAP + 1): c + 1] if c < len(w["cross"]) else np.zeros(0, bool)
    hits = np.flatnonzero(cross)
    if ok.mean() < 0.5:
        f["wind_reversal_7d"], f["wind_days_since_reversal"] = np.nan, np.nan
    else:
        since = (len(cross) - 1 - hits[-1]) if len(hits) else REVERSAL_CAP
        f["wind_days_since_reversal"] = float(min(since, REVERSAL_CAP))
        f["wind_reversal_7d"] = float(since <= 6)
    _CACHE[key] = f
    return f


def nemo_features(cutoff) -> dict:
    key = ("nemo", _day(cutoff))
    if key in _CACHE:
        return _CACHE[key]
    n, c = nemo_daily(), key[1]
    sst3, sst14 = _window(n["sst"], c, 3), _window(n["sst"], c, 14)
    m3, mn14 = _mean(sst3), _stat(sst14, np.min)
    f = {"nemo_sst_rise": m3 - mn14 if np.isfinite(m3) and np.isfinite(mn14) else np.nan,
         "nemo_sst_trend_7d": (m3 - _mean(_window(n["sst"], c - 7, 3)))
         if np.isfinite(m3) else np.nan,
         "nemo_o2_min_7d": _stat(_window(n["o2"], c, 7), np.min),
         "nemo_ph_min_7d": _stat(_window(n["ph"], c, 7), np.min),
         "nemo_chl_3d": _mean(_window(n["chl"], c, 3)),
         "nemo_chl_max_7d": _stat(_window(n["chl"], c, 7), np.max),
         "nemo_coverage": float((~np.isnan(_window(n["sst"], c, 7))).mean())}
    _CACHE[key] = f
    return f


def interaction_features(wind: dict, pn_last_log: float, nb_max_14: float, last_da: float) -> dict:
    rev = wind.get("wind_reversal_7d", np.nan)
    def times(x):
        return rev * x if np.isfinite(rev) and np.isfinite(x) else np.nan
    return {"rev_x_pn": times(pn_last_log),
            "rev_x_nb_max": times(np.log1p(nb_max_14) if np.isfinite(nb_max_14) else np.nan),
            "rev_x_last_da": times(np.log1p(last_da) if np.isfinite(last_da) else np.nan)}


def pn_crash_features(pdates: np.ndarray, pvals: np.ndarray, cutoff) -> dict:
    """H-crash: the site's raw PN samples dated in the 56 days ending at ``cutoff``: the
    peak (log1p), the fractional drop from it to the latest sample, weeks since the peak,
    and a crash flag (drop >= 0.9 from a peak >= 50,000 cells/L)."""
    f = dict.fromkeys(PN_CRASH_COLS, np.nan)
    c = np.datetime64(pd.Timestamp(cutoff), "ns")
    lo = np.searchsorted(pdates, c - np.timedelta64(PN_WINDOW - 1, "D"), side="left")
    hi = np.searchsorted(pdates, c, side="right")
    if hi - lo < 1:
        return f
    d, v = pdates[lo:hi], pvals[lo:hi]
    k = int(np.argmax(v))
    f["pn_peak_8w_log"] = float(np.log1p(v[k]))
    f["pn_weeks_since_peak"] = float((c - d[k]) / np.timedelta64(7, "D"))
    if hi - lo >= 2 and v[k] > 0:
        f["pn_drop_frac"] = float(1 - v[-1] / v[k])
        f["pn_crash"] = float(f["pn_drop_frac"] >= PN_CRASH_DROP and v[k] >= PN_BLOOM)
    return f


def sat_series() -> dict:
    """Per region: (dates, scores) of the committed OAD scores whose valid-pixel fraction
    is >= 0.2, sorted by date."""
    if "sat" not in _DAILY:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        s = pd.read_parquet(os.path.join(root, "data/processed/oad_scores.parquet"))
        c = pd.read_parquet(os.path.join(root, "data/processed/oad_cloud_fractions.parquet"))
        for x in (s, c):
            x["date"] = pd.to_datetime(x["date"])
        m = s.merge(c, on=["date", "region"], how="left").dropna(subset=["score"])
        m = m[m["valid_frac"] >= SAT_MIN_VALID].sort_values("date")
        _DAILY["sat"] = {r: (g["date"].to_numpy("datetime64[ns]"), g["score"].to_numpy(float))
                         for r, g in m.groupby("region")}
    return _DAILY["sat"]


def sat_features(site: str, cutoff) -> dict:
    """H-sat-lag: the source region's valid OAD scores dated <= cutoff - 5 (composite ends
    before the cutoff): max over [cutoff - 56, cutoff - 14], that max minus the latest
    valid score, and the number of valid scores in the span."""
    key = ("sat", SAT_SOURCE[site], _day(cutoff))
    if key in _CACHE:
        return _CACHE[key]
    d, v = sat_series()[SAT_SOURCE[site]]
    c = np.datetime64(pd.Timestamp(cutoff), "ns")
    usable = np.searchsorted(d, c - np.timedelta64(SAT_SHIFT, "D"), side="right")
    lo = np.searchsorted(d, c - np.timedelta64(SAT_SPAN[1], "D"), side="left")
    hi = min(np.searchsorted(d, c - np.timedelta64(SAT_SPAN[0], "D"), side="right"), usable)
    span = v[lo:hi] if hi > lo else v[:0]
    f = {"sat_valid_2_8w": float(len(span)),
         "sat_max_2_8w": float(span.max()) if len(span) else np.nan}
    f["sat_decline"] = (f["sat_max_2_8w"] - float(v[usable - 1])
                        if len(span) and usable > 0 else np.nan)
    _CACHE[key] = f
    return f

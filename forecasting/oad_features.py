"""
OAD (Ocean Anomaly Detection) regional features for DATect.

Joins regional daily anomaly scores produced by the AE_3d_l32_c4_t4_s42_mae050
masked-autoencoder onto DATect's weekly (Monday) per-site feature rows, plus
a parallel valid-pixel-fraction signal (cloud-confound mitigation).

Leakage policy
--------------
DATect rows are weekly (Monday). Engine fetches the row with date <= anchor_date
(= test_date - 7) at predict time. MODIS 8-day composites are CENTERED, so the
score timestamped D contains data from roughly [D-4, D+4]. To guarantee no
composite-window overlap with the forecast horizon (anchor_date, test_date],
we lag the OAD score by an extra LEAK_SHIFT_DAYS = 5 days: every feature anchors
on (row_date - 5) or earlier. Worst-case composite-end = (R-5)+4 = R-1 < R
<= anchor_date < test_date. Safe.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

LEAK_SHIFT_DAYS = 5

# Exact region-name strings as produced by ocean anomaly detection/src/regions.py
# (en-dash in "Overall" is intentional; matches the OAD parquet).
REGION_OLYMPIC = "Olympic Coast (WA)"
REGION_SW_WA = "SW Washington / Long Beach"
REGION_CENTRAL_OR = "Central Oregon"
REGION_SOUTHERN = "Southern OR / N CA"
REGION_OVERALL = "Overall (WA–OR–N. CA coastal)"

SITE_TO_REGION: dict[str, str] = {
    "Kalaloch":       REGION_OLYMPIC,
    "Quinault":       REGION_OLYMPIC,
    "Copalis":        REGION_OLYMPIC,
    "Twin Harbors":   REGION_SW_WA,
    "Long Beach":     REGION_SW_WA,
    "Clatsop Beach":  REGION_SW_WA,
    "Cannon Beach":   REGION_SW_WA,
    "Newport":        REGION_CENTRAL_OR,
    "Coos Bay":       REGION_CENTRAL_OR,
    "Gold Beach":     REGION_SOUTHERN,
}

OAD_FEATURES_LOCAL = [
    "oad_score",
    "oad_score_lag1week",
    "oad_score_lag2week",
    "oad_score_30day_mean",
    "oad_score_30day_max",
    "oad_score_30day_trend",
    "oad_score_zscore_doy",
]
OAD_FEATURES_OVERALL = [f"oad_overall_{n[len('oad_'):]}" for n in OAD_FEATURES_LOCAL]

OAD_CLOUD_FEATURES = [
    "oad_valid_frac",
    "oad_overall_valid_frac",
]

OAD_FEATURES_ALL = OAD_FEATURES_LOCAL + OAD_FEATURES_OVERALL + OAD_CLOUD_FEATURES


def compute_region_features(
    region_scores: pd.DataFrame,
    row_dates,
    *,
    feature_prefix: str = "oad_",
) -> pd.DataFrame:
    """Compute 7 OAD-derived features for each row date, against one region's daily series."""
    s = (
        region_scores.dropna(subset=["date", "score"])
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .drop_duplicates("date")
        .set_index("date")
        .sort_index()["score"]
    )

    cols = [
        "score", "score_lag1week", "score_lag2week",
        "score_30day_mean", "score_30day_max", "score_30day_trend",
        "score_zscore_doy",
    ]
    row_dates = pd.DatetimeIndex(pd.to_datetime(row_dates))
    out = pd.DataFrame(index=range(len(row_dates)), columns=cols, dtype=float)

    s_df = s.reset_index()
    s_df["year"] = s_df["date"].dt.year
    s_df["doy"] = s_df["date"].dt.dayofyear

    def _window_mean(end_date, days=7):
        start = end_date - pd.Timedelta(days=days - 1)
        w = s.loc[(s.index >= start) & (s.index <= end_date)]
        return float(w.mean()) if len(w) > 0 else np.nan

    for i, R in enumerate(row_dates):
        R = pd.Timestamp(R)
        d_anchor = R - pd.Timedelta(days=LEAK_SHIFT_DAYS)        # R - 5
        d_lag1 = R - pd.Timedelta(days=LEAK_SHIFT_DAYS + 7)      # R - 12
        d_lag2 = R - pd.Timedelta(days=LEAK_SHIFT_DAYS + 14)     # R - 19

        out.at[i, "score"] = _window_mean(d_anchor, 7)
        out.at[i, "score_lag1week"] = _window_mean(d_lag1, 7)
        out.at[i, "score_lag2week"] = _window_mean(d_lag2, 7)

        win = s.loc[(s.index >= R - pd.Timedelta(days=35)) & (s.index <= d_anchor)]
        if len(win) >= 5:
            out.at[i, "score_30day_mean"] = float(win.mean())
            out.at[i, "score_30day_max"] = float(win.max())
            x = (win.index - win.index[0]).days.values.astype(float)
            y = win.values.astype(float)
            if x.std() > 0:
                out.at[i, "score_30day_trend"] = float(np.polyfit(x, y, 1)[0])

        target_doy = R.dayofyear
        target_year = R.year
        prior_years_mask = s_df["year"] < target_year
        if prior_years_mask.any():
            doy_diff = (s_df["doy"] - target_doy).abs()
            doy_diff = np.minimum(doy_diff, 365 - doy_diff)
            climo_mask = prior_years_mask & (doy_diff <= 16)
            climo_vals = s_df.loc[climo_mask, "score"]
            if len(climo_vals) >= 10 and climo_vals.std() > 0:
                anchor_val = out.at[i, "score"]
                if not pd.isna(anchor_val):
                    out.at[i, "score_zscore_doy"] = (
                        (anchor_val - climo_vals.mean()) / climo_vals.std()
                    )

    out.columns = [feature_prefix + c for c in cols]
    return out


def add_oad_features(
    compiled_df: pd.DataFrame,
    oad_parquet_path: str,
    *,
    cloud_parquet_path: str | None = None,
    date_col: str = "Date",
    site_col: str = "Site",
) -> pd.DataFrame:
    """Left-join 16 OAD-derived features onto compiled_df (14 score + 2 cloud)."""
    oad = pd.read_parquet(oad_parquet_path)
    oad["date"] = pd.to_datetime(oad["date"])

    if "aggregation" in oad.columns and oad["aggregation"].nunique() > 1:
        oad = oad[oad["aggregation"] == "mean"].copy()

    out = compiled_df.copy().reset_index(drop=True)
    out[date_col] = pd.to_datetime(out[date_col])

    # Only validate regions actually needed for sites present in compiled_df
    sites_present = set(out[site_col].unique())
    needed_regions = {SITE_TO_REGION[s] for s in sites_present if s in SITE_TO_REGION}
    needed_regions.add(REGION_OVERALL)
    available = set(oad["region"].unique())
    missing = needed_regions - available
    if missing:
        raise ValueError(f"OAD parquet missing required regions: {missing}")

    region_scores: dict[str, pd.DataFrame] = {
        r: oad.loc[oad["region"] == r, ["date", "score"]].copy()
        for r in needed_regions
    }

    # Local-region features per site
    local_pieces = []
    for site, region in SITE_TO_REGION.items():
        mask = out[site_col] == site
        if not mask.any():
            continue
        site_rows = out.loc[mask]
        feats = compute_region_features(
            region_scores[region], site_rows[date_col].values, feature_prefix="oad_"
        )
        feats.index = site_rows.index
        local_pieces.append(feats)
    local_df = pd.concat(local_pieces).sort_index()

    # Overall-region features: per unique date, broadcast back
    unique_dates = pd.DatetimeIndex(out[date_col].drop_duplicates().sort_values())
    overall_unique = compute_region_features(
        region_scores[REGION_OVERALL], unique_dates, feature_prefix="oad_overall_"
    )
    overall_unique.index = unique_dates
    overall_df = overall_unique.loc[out[date_col].values].reset_index(drop=True)
    overall_df.index = out.index

    # Cloud features
    cloud_local = pd.Series(np.nan, index=out.index, name="oad_valid_frac")
    cloud_overall = pd.Series(np.nan, index=out.index, name="oad_overall_valid_frac")
    if cloud_parquet_path and os.path.exists(cloud_parquet_path):
        cloud = pd.read_parquet(cloud_parquet_path)
        cloud["date"] = pd.to_datetime(cloud["date"])
        cloud_by_region = {
            r: cloud.loc[cloud["region"] == r].set_index("date")["valid_frac"].sort_index()
            for r in cloud["region"].unique()
        }

        def _cloud_7day(region_name, R):
            s = cloud_by_region.get(region_name)
            if s is None:
                return np.nan
            d_anchor = R - pd.Timedelta(days=LEAK_SHIFT_DAYS)
            w = s.loc[(s.index >= d_anchor - pd.Timedelta(days=6)) & (s.index <= d_anchor)]
            return float(w.mean()) if len(w) > 0 else np.nan

        for idx in out.index:
            site = out.at[idx, site_col]
            R = out.at[idx, date_col]
            cloud_local.at[idx] = _cloud_7day(SITE_TO_REGION[site], R)
            cloud_overall.at[idx] = _cloud_7day(REGION_OVERALL, R)

    return pd.concat([out, local_df, overall_df, cloud_local, cloud_overall], axis=1)

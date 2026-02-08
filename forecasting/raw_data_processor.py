"""
Minimal RawDataProcessor for raw-DA forecasting.

Creates observation-order lag features instead of grid-position lags,
which is critical for sparse/irregular raw DA measurement data.
"""

import pandas as pd
import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


class RawDataProcessor:
    """
    Minimal processor to create raw lag features without interpolation.
    """

    def create_raw_lag_features(self, df, group_col, value_col, lags):
        """
        Create observation-order lag features instead of grid-position lags.

        For each row, the Nth lag is the Nth most recent ACTUAL observation
        (non-NaN value_col) strictly before this row's date.  This avoids the
        problem of grid-shift lags being mostly NaN on sparse/irregular data.

        Features created (where N = min(max(lags), 4)):
          {value_col}_prev_obs_1  ..  _prev_obs_N   : observed values
          {value_col}_prev_obs_2_weeks_ago .. _N_weeks_ago : recency
          {value_col}_prev_obs_diff_1_2 : trend between two most recent obs
        """
        max_obs_lags = min(max(lags), 4) if lags else 0
        if max_obs_lags == 0:
            return df

        logger.info(
            "Creating %d observation-order lag features for %s",
            max_obs_lags,
            value_col,
        )
        df = df.copy()
        df = df.sort_values([group_col, "date"]).reset_index(drop=True)

        # Initialise new columns
        for i in range(1, max_obs_lags + 1):
            df[f"{value_col}_prev_obs_{i}"] = np.nan
            if i >= 2:
                df[f"{value_col}_prev_obs_{i}_weeks_ago"] = np.nan

        # Process each site
        for site, site_idx in df.groupby(group_col).groups.items():
            site_df = df.loc[site_idx]

            # Actual observations (non-NaN) for this site
            obs_mask = site_df[value_col].notna()
            obs_dates = site_df.loc[obs_mask, "date"].values
            obs_values = site_df.loc[obs_mask, value_col].values

            if len(obs_values) == 0:
                continue

            for idx in site_idx:
                row_date = df.at[idx, "date"]
                # Observations strictly before this row
                past_mask = obs_dates < row_date
                n_past = past_mask.sum()
                if n_past == 0:
                    continue

                past_vals = obs_values[past_mask]
                past_dts = obs_dates[past_mask]

                for lag_i in range(1, min(max_obs_lags + 1, n_past + 1)):
                    val = float(past_vals[-lag_i])
                    df.at[idx, f"{value_col}_prev_obs_{lag_i}"] = val
                    if lag_i >= 2:
                        obs_dt = pd.Timestamp(past_dts[-lag_i])
                        weeks_ago = (row_date - obs_dt).days / 7.0
                        df.at[idx, f"{value_col}_prev_obs_{lag_i}_weeks_ago"] = weeks_ago

        # Derived: difference between two most recent observations
        if max_obs_lags >= 2:
            df[f"{value_col}_prev_obs_diff_1_2"] = (
                df[f"{value_col}_prev_obs_1"] - df[f"{value_col}_prev_obs_2"]
            )

        return df

"""
Base interface for feature-extension chains.

A "chain" is one experimental feature group (e.g., lagged PN, BEUTI derivatives,
NEMO mooring features). Each chain module exports:

    CHAIN_NAME       : str
    NEW_FEATURES     : list[str]    # column names that will be added to the parquet
    TARGET_SITES     : list[str]    # which DATect sites should get these features
                                    # in their feature_subset (None = all sites)
    LEAK_SHIFT_DAYS  : int          # leak-safe lag (matches OAD's 5 by default)

    def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
        '''Return df_in with new feature columns appended. df_in is the
        existing final_output.parquet (rows = site×weekly-date).
        Implementation MUST honor LEAK_SHIFT_DAYS so features at row R
        are computed only from data with date <= R - LEAK_SHIFT_DAYS.'''

The runner `chains/run_chain.py` handles:
  1. Loading data/processed/final_output.parquet
  2. Calling chain.add_features() to produce augmented df
  3. Writing chains/output/final_output_<name>.parquet
  4. Running an A/B retrospective (with vs without those columns)
  5. Reporting Δ R² per site + pooled
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class Chain(Protocol):
    CHAIN_NAME: str
    NEW_FEATURES: list[str]
    TARGET_SITES: list[str] | None
    LEAK_SHIFT_DAYS: int

    def add_features(self, df_in: pd.DataFrame) -> pd.DataFrame: ...

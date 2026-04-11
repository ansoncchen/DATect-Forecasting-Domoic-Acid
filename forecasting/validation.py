"""
Configuration and System Validation Module - Simplified
======================================================

Provides essential validation functions for system startup.
"""

import config
import pandas as pd


def validate_system_startup():
    """Essential system startup validation."""
    # Check critical configuration
    if not hasattr(config, 'FINAL_OUTPUT_PATH') or not config.FINAL_OUTPUT_PATH:
        raise ValueError("Missing FINAL_OUTPUT_PATH configuration")
    if not hasattr(config, 'SITES') or not config.SITES:
        raise ValueError("No sites configured in SITES dictionary")
    return True


def validate_runtime_parameters(n_anchors=None, min_test_date=None):
    """Validate retrospective analysis parameters."""
    if n_anchors is not None and (not isinstance(n_anchors, int) or n_anchors < 1):
        raise ValueError(f"n_anchors must be a positive integer, got: {n_anchors}")
    if min_test_date is not None:
        if not isinstance(min_test_date, str):
            raise TypeError(
                f"min_test_date must be str or None, got {type(min_test_date).__name__}"
            )
        parsed = pd.Timestamp(min_test_date)
        if pd.isna(parsed):
            raise ValueError(f"min_test_date could not be parsed as a date: {min_test_date!r}")
    return True
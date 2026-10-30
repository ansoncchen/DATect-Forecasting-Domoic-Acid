"""
Configuration and System Validation Module - Simplified
======================================================

Provides essential validation functions for system startup.
"""

import config


def validate_system_startup():
    """Essential system startup validation."""
    # Check critical configuration
    if not hasattr(config, 'FINAL_OUTPUT_PATH') or not config.FINAL_OUTPUT_PATH:
        raise ValueError("Missing FINAL_OUTPUT_PATH configuration")
    if not hasattr(config, 'SITES') or not config.SITES:
        raise ValueError("No sites configured in SITES dictionary")
    return True

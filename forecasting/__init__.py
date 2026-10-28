"""
Core Forecasting Components
===========================

This module contains the core logic for domoic acid forecasting:

- ForecastEngine: Raw-data forecasting engine (XGBoost + RF ensemble; naïve persistence as a separate baseline)
- ModelFactory: Ensemble model factory with per-site configuration

The public names are resolved lazily so that ``config.py`` can import
``forecasting.tuned_config`` without loading the engine (which itself imports
``config`` and would otherwise see a half-initialised module).
"""

__all__ = ['ForecastEngine', 'ModelFactory']


def __getattr__(name):
    if name == 'ForecastEngine':
        from .raw_forecast_engine import RawForecastEngine
        return RawForecastEngine
    if name == 'ModelFactory':
        from .ensemble_model_factory import EnsembleModelFactory
        return EnsembleModelFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

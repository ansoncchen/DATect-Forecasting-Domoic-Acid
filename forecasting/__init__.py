"""
Core Forecasting Components
===========================

This module contains the core logic for domoic acid forecasting:

- ForecastEngine: Raw-data forecasting engine (XGBoost + RF ensemble; naïve persistence as a separate baseline)
- ModelFactory: Ensemble model factory with per-site configuration
"""

from .raw_forecast_engine import RawForecastEngine as ForecastEngine
from .ensemble_model_factory import EnsembleModelFactory as ModelFactory

__all__ = ['ForecastEngine', 'ModelFactory']

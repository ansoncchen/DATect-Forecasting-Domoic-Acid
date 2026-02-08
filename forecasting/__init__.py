"""
Core Forecasting Components
===========================

This module contains the core logic for domoic acid forecasting:

- ForecastEngine: Raw-data ensemble forecasting engine (XGBoost + RF + Naive)
- DataProcessor: Data cleaning and feature engineering (used by visualizations)
- ModelFactory: Ensemble model factory with per-site configuration

The raw-data pipeline replaces the original interpolated-data pipeline
while preserving the same public API for the FastAPI backend.
"""

# Primary engine: raw-data ensemble pipeline
from .raw_forecast_engine import RawForecastEngine as ForecastEngine

# Keep original DataProcessor for visualizations and other utilities
from .data_processor import DataProcessor

# Ensemble-aware model factory
from .ensemble_model_factory import EnsembleModelFactory as ModelFactory

__all__ = ['ForecastEngine', 'DataProcessor', 'ModelFactory']

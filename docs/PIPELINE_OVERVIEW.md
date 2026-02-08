# DATect Pipeline Overview

## System Architecture

The DATect forecasting system processes environmental data to predict domoic acid concentrations. This document provides a high-level overview of the pipeline.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION STAGE                          │
│  (dataset-creation.py - Run once, takes 30-60 minutes)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Raw Data Sources                        │
        │  • MODIS Satellite (8-day composites)   │
        │  • Climate Indices (PDO, ONI, BEUTI)    │
        │  • Streamflow (USGS Columbia River)     │
        │  • DA Measurements (10 sites)           │
        │  • Pseudo-nitzschia Cell Counts         │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Data Processing                         │
        │  • Download & spatial aggregation       │
        │  • Temporal alignment (weekly grid)     │
        │  • Apply temporal buffers:              │
        │    - 7-day satellite delay              │
        │    - 2-month climate index delay        │
        │  • Biological decay interpolation       │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Output: final_output.parquet           │
        │  Unified weekly time series with:       │
        │  • All environmental features           │
        │  • DA & PN measurements                 │
        │  • Site coordinates                     │
        └─────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    FORECASTING STAGE                             │
│  (forecasting/raw_forecast_engine.py - Per request)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Data Loading & Preparation              │
        │  (forecasting/raw_data_forecaster.py)   │
        │  • Load final_output.parquet            │
        │  • Build observation-order lag features  │
        │  • Add temporal features (sin/cos)      │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Temporal Validation                     │
        │  (config.verify_no_data_leakage)        │
        │  • Calculate anchor date                │
        │  • Verify no future data leakage        │
        │  • Create per-forecast DA categories    │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Ensemble Training & Prediction          │
        │  (forecasting/raw_forecast_engine.py)   │
        │  • Train/test split at anchor date      │
        │  • Per-site XGB + RF + Naive ensemble   │
        │  • Weighted blending via per_site_models │
        │  • Generate prediction                  │
        │  • Quantile/bootstrap intervals         │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Results                                 │
        │  • Predicted DA concentration           │
        │  • Risk category (if classification)    │
        │  • Confidence intervals (5%, 50%, 95%)  │
        │  • Feature importance rankings          │
        └─────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    WEB INTERFACE                                 │
│  (backend/api.py + frontend/)                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Dashboard & Analysis Tools             │
        │  • Single forecasts                     │
        │  • Retrospective evaluation             │
        │  • Historical analysis                  │
        │  • Visualizations                       │
        └─────────────────────────────────────────┘
```

## Key Components

### Data Ingestion (`dataset-creation.py`)

Downloads and processes environmental data:

| Data Source | Variables | Temporal Buffer |
|-------------|-----------|-----------------|
| MODIS Satellite | CHL, SST, PAR, FLH, K490 | 7 days |
| Monthly Anomalies | CHLA-anom, SST-anom | 2 months |
| Climate Indices | PDO, ONI, BEUTI | 2 months |
| Streamflow | Columbia River discharge | None (as-of merge) |
| Biological | DA, Pseudo-nitzschia | Decay interpolation |

**Output**: `data/processed/final_output.parquet` (weekly time series, 2003-2023)

### Forecasting Engine (`forecasting/raw_forecast_engine.py`)

Generates predictions with the 3-model ensemble:

1. **Anchor Date Calculation**: `forecast_date - FORECAST_HORIZON_DAYS`
2. **Observation-Order Lag Features**: Past-only shifts on raw DA measurements
3. **Train/Test Split**: Chronological (training ≤ anchor_date)
4. **DA Category Creation**: Per-forecast from training data only
5. **Ensemble Training**: XGBoost + Random Forest + Naive with per-site weights
6. **Prediction**: With configurable confidence intervals

### Per-Site Configuration (`forecasting/per_site_models.py`)

Custom tuning for each of the 10 monitoring sites:

- XGBoost/RF hyperparameters per site
- Feature subsets per site
- Ensemble weights per site
- Prediction clipping per site

### Data Pipeline (`forecasting/raw_data_forecaster.py`, `forecasting/raw_data_processor.py`)

Handles feature engineering for raw measurements:

- Observation-order lag features (not grid-shift)
- Temporal encoding (sin/cos day-of-year)
- Rolling statistics (mean/std over multiple windows)
- Leak-free test row construction

### Web API (`backend/api.py`)

REST endpoints:

| Endpoint | Description |
|----------|-------------|
| `POST /api/forecast` | Single forecast |
| `POST /api/forecast/enhanced` | Regression + classification |
| `POST /api/retrospective` | Run retrospective evaluation |
| `GET /api/historical/{site}` | Historical DA data |
| `GET /api/visualizations/*` | Analysis plots |

## Configuration (`config.py`)

Key settings:

```python
# Forecast horizon
FORECAST_HORIZON_WEEKS = 1
FORECAST_HORIZON_DAYS = 7

# Models
FORECAST_MODEL = "ensemble"   # "ensemble", "naive", or "linear"
FORECAST_TASK = "regression"

# Features
USE_LAG_FEATURES = True
USE_ROLLING_FEATURES = True
USE_ENHANCED_TEMPORAL_FEATURES = True
USE_PER_SITE_MODELS = True

# Evaluation
N_RANDOM_ANCHORS = 500
N_BOOTSTRAP_ITERATIONS = 20

# DA categories (μg/g)
DA_CATEGORY_BINS = [-inf, 5, 20, 40, inf]  # Low, Moderate, High, Extreme
```

## Data Flow Summary

```
Raw Data → dataset-creation.py → final_output.parquet
                                        ↓
                              forecasting/raw_data_forecaster.py
                                        ↓
                              forecasting/raw_forecast_engine.py
                                        ↓
                              backend/api.py → JSON responses
                                        ↓
                              frontend/ → User interface
```

## Running the Pipeline

### Initial Setup

```bash
# Generate dataset (required once, 30-60 min)
python dataset-creation.py
```

### Daily Operations

```bash
# Start web application
python run_datect.py
# Opens browser at http://localhost:3000
```

### Validation

```bash
# Temporal validation runs automatically during cache generation
python precompute_cache.py
```

### Programmatic Use

```python
from forecasting import ForecastEngine

engine = ForecastEngine()

# Single forecast
result = engine.generate_single_forecast(
    data_path="data/processed/final_output.parquet",
    forecast_date="2020-06-15",
    site="Newport",
    task="regression",
    model_type="ensemble"
)

# Retrospective evaluation
results_df = engine.run_retrospective_evaluation(
    task="regression",
    model_type="ensemble",
    n_anchors=500
)
```

## Temporal Safeguards

| Safeguard | Purpose |
|-----------|---------|
| Chronological split | Training always before test |
| Temporal buffer | Minimum gap between train/test |
| Satellite delay | 7-day processing buffer |
| Climate delay | 2-month reporting buffer |
| Lag cutoffs | No future data in features |
| Per-forecast categories | No target leakage |

## File Structure

```
DATect-Forecasting-Domoic-Acid/
├── dataset-creation.py           # Data ingestion pipeline
├── run_datect.py                 # Application launcher
├── config.py                     # Configuration
├── precompute_cache.py           # Cache generation with validation
├── forecasting/
│   ├── raw_forecast_engine.py    # Ensemble pipeline (XGB + RF + Naive)
│   ├── raw_data_forecaster.py    # Raw DA loading, feature frame building
│   ├── raw_data_processor.py     # Observation-order lag features
│   ├── per_site_models.py        # Per-site hyperparams and ensemble weights
│   ├── ensemble_model_factory.py # Ensemble model wrapper
│   ├── classification_adapter.py # DA risk category classification
│   ├── feature_utils.py          # Shared temporal feature engineering
│   └── validation.py             # System startup validation
├── backend/
│   ├── api.py                    # FastAPI endpoints
│   ├── visualizations.py         # Plot generation
│   └── cache_manager.py          # Result caching
├── frontend/                     # React web interface
└── data/
    └── processed/
        └── final_output.parquet  # Unified dataset
```

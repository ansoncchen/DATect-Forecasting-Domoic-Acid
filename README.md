# DATect - Domoic Acid Forecasting System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React-blue.svg)](https://reactjs.org/)

## Overview

DATect is a machine learning system for forecasting harmful algal bloom toxin concentrations (domoic acid) along the Pacific Coast. The system integrates satellite oceanographic data, climate indices, and environmental measurements with strict temporal safeguards to ensure scientifically valid predictions.

**Key Features:**
- 10 monitoring sites from Oregon to Washington
- 21 years of integrated data (2003-2023)
- XGBoost and linear model options
- Comprehensive temporal validation (7 integrity tests)
- Bootstrap confidence intervals for uncertainty quantification

## Quick Start

```bash
# Clone repository
git clone https://github.com/your-username/DATect-Forecasting-Domoic-Acid.git
cd DATect-Forecasting-Domoic-Acid

# Run locally (auto-installs dependencies)
python run_datect.py
```

Opens at http://localhost:3000

## Commands

| Command | Description |
|---------|-------------|
| `python run_datect.py` | Start system (backend + frontend) |
| `python precompute_cache.py` | Pre-compute cache and validate |
| `python dataset-creation.py` | Regenerate dataset (30-60 min) |
| `./deploy_gcloud.sh` | Deploy to Google Cloud |

## System Architecture

```
DATect-Forecasting-Domoic-Acid/
├── run_datect.py           # System launcher
├── dataset-creation.py     # Data pipeline (satellite, climate, toxins)
├── config.py               # Configuration (sites, models, parameters)
├── forecasting/            # ML engine
│   ├── forecast_engine.py  # Core forecasting with temporal safeguards
│   ├── data_processor.py   # Feature engineering
│   ├── model_factory.py    # XGBoost/linear model creation
│   └── validation.py       # Temporal integrity checks
├── backend/                # FastAPI server
│   ├── api.py              # REST endpoints
│   ├── visualizations.py   # Chart generation
│   └── cache_manager.py    # Result caching
├── frontend/               # React + Vite interface
└── data/processed/         # Processed dataset (parquet)
```

## Using the Dashboard

1. **Select date** (2008-2024 range)
2. **Select site** (10 Pacific Coast locations)
3. **Select model** (XGBoost recommended)
4. **Click "Forecast"**

**Risk Categories:**
- **Low** (≤5 μg/g): Safe for consumption
- **Moderate** (5-20 μg/g): Caution advised
- **High** (20-40 μg/g): Avoid consumption (above federal limit)
- **Extreme** (>40 μg/g): Health hazard

**Retrospective Mode:** Compare XGBoost vs linear model performance across 500 anchor points per site.

## Data Sources

- **Satellite**: MODIS chlorophyll-a, SST, PAR, fluorescence, K490 (8-day composites)
- **Climate Indices**: PDO, ONI, BEUTI (2-month reporting delay enforced)
- **Streamflow**: USGS Columbia River discharge
- **Toxin Data**: State monitoring programs (WA, OR)

## Documentation

- [Forecast Pipeline](docs/FORECAST_PIPELINE.md) - Technical data flow
- [Pipeline Overview](docs/PIPELINE_OVERVIEW.md) - System architecture
- [Scientific Validation](docs/SCIENTIFIC_VALIDATION.md) - Temporal safeguards
- [Visualizations Guide](docs/VISUALIZATIONS_GUIDE.md) - Chart interpretation
- [Quick Start](docs/QUICK_START.md) - Setup instructions

## Temporal Safeguards

The system enforces strict temporal integrity to prevent data leakage:

1. **Chronological splits**: Training data always precedes test data
2. **Satellite buffer**: 7-day processing delay
3. **Climate buffer**: 2-month reporting delay
4. **Per-forecast categories**: DA risk levels computed from training data only
5. **Lag feature cutoffs**: No future data in feature calculations

## Google Cloud Deployment

```bash
gcloud auth login
gcloud config set project YOUR-PROJECT-ID
./deploy_gcloud.sh
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port in use | `run_datect.py` auto-kills existing processes |
| Missing dataset | Run `python dataset-creation.py` |
| Node.js not found | Install from [nodejs.org](https://nodejs.org/) |

## License

Scientific research project. Please cite if used in publications.

## Acknowledgments

- NOAA CoastWatch for satellite data
- USGS for streamflow data
- Olympic Region HAB Partnership, WA DOH, OR DFW for toxin measurements

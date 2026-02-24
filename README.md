# DATect — Domoic Acid Forecasting System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React-blue.svg)](https://reactjs.org/)

## Overview

DATect is a machine learning system for forecasting harmful algal bloom toxin concentrations (domoic acid) along the Pacific Coast. It integrates satellite oceanographic data, climate indices, and environmental measurements with strict temporal safeguards to ensure scientifically valid predictions.

**Key features:**
- 10 monitoring sites from Oregon to Washington
- 21 years of integrated data (2003–2023)
- 3-model ensemble (XGBoost + Random Forest + Naive) with per-site hyperparameter tuning
- Observation-order lag features for sparse/irregular measurement data
- No-data-leakage guarantees — `verify_no_data_leakage()` called on every prediction
- Quantile regression + bootstrap confidence intervals

## Quick Start

```bash
# Clone repository
git clone https://github.com/ansoncchen/DATect-Forecasting-Domoic-Acid.git
cd DATect-Forecasting-Domoic-Acid

# Run locally (auto-installs dependencies)
python3 run_datect.py
```

Opens at http://localhost:3000. Backend API at http://localhost:8000 | API docs at http://localhost:8000/docs.

Auto-detects `uv` (10× faster installs), `bun` (faster frontend), and `granian` (20–40% faster ASGI) if installed.

## Commands

| Command | Description |
|---------|-------------|
| `python3 run_datect.py` | Start system (backend + frontend + browser) |
| `python3 precompute_cache.py` | Pre-compute cache — **run on Hyak, not locally** |
| `python3 dataset-creation.py` | Regenerate dataset (30–60 min, only when data changes) |
| `./deploy_gcloud.sh` | Deploy to Google Cloud Run |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENABLE_PRECOMPUTED_CACHE` | `false` | Set `"true"` to use pre-computed cache locally |
| `CACHE_DIR` | `./cache` | Path to pre-computed cache directory |
| `REDIS_URL` | unset | Redis URL for 100× faster cache reads (e.g. `redis://localhost:6379/0`) |

## System Architecture

```
DATect-Forecasting-Domoic-Acid/
├── run_datect.py                   # System launcher (auto-detects uv/bun/granian)
├── dataset-creation.py             # Data pipeline (satellite, climate, toxins)
├── precompute_cache.py             # Cache pre-computation (run on Hyak)
├── config.py                       # All configuration (sites, models, parameters)
├── forecasting/                    # ML forecasting engine
│   ├── raw_forecast_engine.py      # Main engine — ensemble pipeline with per-site tuning
│   ├── raw_data_forecaster.py      # Raw DA loading, feature frame building, leak-free test rows
│   ├── raw_data_processor.py       # Observation-order lag features (not grid-shift)
│   ├── per_site_models.py          # Per-site hyperparams, feature subsets, ensemble weights
│   ├── ensemble_model_factory.py   # Class-based wrapper matching API's ModelFactory interface
│   ├── raw_model_factory.py        # Standalone model builders (XGB, RF, classifier)
│   ├── classification_adapter.py   # Threshold + dedicated XGBoost classifier for 4 DA categories
│   ├── feature_utils.py            # Shared temporal features + transformer creation
│   ├── logging_config.py           # Logging configuration
│   └── validation.py               # System startup validation
├── backend/                        # FastAPI web server
│   ├── api.py                      # REST endpoints
│   ├── visualizations.py           # Chart generation (correlation, time series, spectral)
│   ├── cache_manager.py            # Pre-computed cache access (file-based + Redis)
│   └── redis_cache.py              # Optional Redis caching (set REDIS_URL to enable)
├── frontend/                       # React + Vite dashboard
├── scripts/
│   └── setup_fast.sh               # Fast environment setup script
├── data/processed/                 # Processed dataset (final_output.parquet)
├── Dockerfile.production           # Production container
└── cloudbuild.yaml                 # Google Cloud Build config
```

## Using the Dashboard

1. **Select date** (2003–2023 range)
2. **Select site** (10 Pacific Coast locations)
3. **Select model** (`ensemble` recommended; also: `naive`, `linear`)
4. **Click "Forecast"**

**Risk categories:**

| Category | DA Level (µg/g) | Meaning |
|----------|-----------------|---------|
| Low | 0–5 | Safe for consumption |
| Moderate | 5–20 | Caution advised |
| High | 20–40 | Avoid (above FDA action level) |
| Extreme | >40 | Health hazard |

**Retrospective mode:** Evaluate model performance across historical anchor points per site.

## Model Performance

| Metric | Value |
|--------|-------|
| Ensemble R² | ~0.49 |
| Ensemble MAE | ~5.4 µg/g |
| Spike F1 | ~0.68 |

## Data Sources

- **Satellite**: MODIS-Aqua chlorophyll-a, SST, PAR, fluorescence (FLH), K490 (8-day composites)
- **Climate indices**: PDO, ONI, BEUTI (2-month reporting delay enforced)
- **Streamflow**: USGS Columbia River discharge (gauge 14246900)
- **Toxin data**: WA DOH and OR DFW state monitoring programs

## Documentation

| Document | Description |
|----------|-------------|
| [docs/FORECAST_PIPELINE.md](docs/FORECAST_PIPELINE.md) | Technical data flow, stages, and API endpoints |
| [docs/PIPELINE_OVERVIEW.md](docs/PIPELINE_OVERVIEW.md) | High-level architecture diagram |
| [docs/PIPELINE_DEEP_DIVE.md](docs/PIPELINE_DEEP_DIVE.md) | Complete step-by-step walkthrough of one prediction |
| [docs/SCIENTIFIC_VALIDATION.md](docs/SCIENTIFIC_VALIDATION.md) | Temporal safeguards and leakage prevention |
| [docs/DATA_PIPELINE_DETAILED.md](docs/DATA_PIPELINE_DETAILED.md) | dataset-creation.py technical details |
| [docs/dataset-creation-scientific-decisions.md](docs/dataset-creation-scientific-decisions.md) | Scientific rationale for data decisions |
| [docs/VISUALIZATIONS_GUIDE.md](docs/VISUALIZATIONS_GUIDE.md) | Chart and visualization interpretation |
| [docs/QUICK_START.md](docs/QUICK_START.md) | OS-specific setup and GCP deployment |
| [docs/HYAK_SETUP.md](docs/HYAK_SETUP.md) | Hyak (UW Klone) cluster workflow for cache pre-computation |

## Temporal Safeguards

The system enforces strict temporal integrity to prevent data leakage:

1. **Chronological splits** — training only uses `date <= anchor_date`
2. **Satellite buffer** — 7-day processing delay
3. **Climate buffer** — 2-month reporting delay
4. **Observation-order lags** — past-only shifts on raw measurements
5. **Per-forecast categories** — DA risk levels computed from training data only
6. **Persistence features recomputed** — from training data only, not global forward-fill
7. **Fresh model per test point** — no lookahead via shared state
8. **`verify_no_data_leakage()`** — called for every prediction, raises `AssertionError` on violation

## Hyak Workflow

Heavy computation (pre-computing the retrospective cache) must run on the Hyak cluster:

```bash
# On Hyak compute node:
python3 precompute_cache.py

# Transfer cache to local machine:
scp -r klone-node:/gscratch/stf/YOUR_UWNETID/DATect-Forecasting-Domoic-Acid/cache/ ./cache/

# Run dashboard locally with pre-computed cache:
ENABLE_PRECOMPUTED_CACHE=true python3 run_datect.py
```

See [docs/HYAK_SETUP.md](docs/HYAK_SETUP.md) for full SSH and environment setup.

## Google Cloud Deployment

```bash
gcloud auth login
gcloud config set project YOUR-PROJECT-ID
./deploy_gcloud.sh
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | `run_datect.py` auto-kills existing processes on 8000/3000 |
| Missing dataset | Run `python3 dataset-creation.py` (30–60 min) |
| Node.js not found | Install from [nodejs.org](https://nodejs.org/) |
| `precompute_cache.py` is slow | Run on Hyak, not locally — see docs/HYAK_SETUP.md |

## License

Scientific research project. Please cite if used in publications.

## Acknowledgments

- NOAA CoastWatch for satellite data (MODIS-Aqua via ERDDAP)
- USGS for Columbia River streamflow data
- Olympic Region HAB Partnership, WA DOH, OR DFW for toxin measurements

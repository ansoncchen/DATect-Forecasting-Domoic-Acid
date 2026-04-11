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
- Two-model ML ensemble (XGBoost + Random Forest) with per-site hyperparameter tuning
- Observation-order lag features for sparse/irregular measurement data
- No-data-leakage guarantees — `_verify_no_data_leakage()` runs on every prediction (see `forecasting/raw_forecast_engine.py`)
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

`run_datect.py` picks the fastest tools present: `uv` or `pip` for Python (`requirements.txt`), `bun` or `npm` for the frontend, `granian` or `uvicorn` for the API.

## Commands

| Command | Description |
|---------|-------------|
| `python3 run_datect.py` | Start system (backend + frontend + browser) |
| `python3 precompute_cache.py` | Pre-compute cache — **run on Hyak, not locally** |
| `python3 dataset-creation.py` | Regenerate dataset (30–60 min, only when data changes) |
| `bash run_full_validation.sh` | Parallel paper metrics, ablation, stability, spike eval, cache (Hyak) |
| `./deploy_gcloud.sh` | Deploy to Google Cloud Run |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENABLE_PRECOMPUTED_CACHE` | `false` | Set `"true"` to use pre-computed cache locally |
| `CACHE_DIR` | `./cache` | Path to pre-computed cache directory |
| `REDIS_URL` | unset | Redis URL for 100× faster cache reads (e.g. `redis://localhost:6379/0`) |
| `ALLOWED_ORIGINS` | unset | Comma-separated CORS origins for production (defaults allow localhost dev only) |

## System Architecture

```
DATect-Forecasting-Domoic-Acid/
├── run_datect.py                   # System launcher (uv/bun/granian when available)
├── dataset-creation.py             # Data pipeline (satellite, climate, toxins)
├── precompute_cache.py             # Cache pre-computation (run on Hyak)
├── run_full_validation.sh          # Parallel paper / cache jobs (Hyak)
├── scripts/eval/                   # Paper metrics, ablation, stability, spike eval
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

Reference numbers match the paper header in `paper/datect_paper_mdpi.tex` (seed=123 test sample, unless noted).

| Metric | Value |
|--------|-------|
| Ensemble R² | 0.215 (independent test, seed=123, 40% sample) / 0.414 (dev, seed=42) / 0.315 (temporal holdout 2019+) |
| Ensemble MAE | 6.42 µg/g (test sample above) |
| Spike / alert | Regression spike recall 0.558 (ensemble); hybrid alert event recall 0.859, transition recall 0.734 |

## Data Sources

- **Satellite**: MODIS-Aqua chlorophyll-a, SST, PAR, fluorescence (FLH), K490 (8-day composites)
- **Climate indices**: PDO, ONI, BEUTI (2-month reporting delay enforced)
- **Streamflow**: USGS Columbia River discharge (gauge 14246900)
- **Toxin data**: WA DOH and OR DFW state monitoring programs

## Documentation

| Document | Description |
|----------|-------------|
| [docs/PIPELINE_DEEP_DIVE.md](docs/PIPELINE_DEEP_DIVE.md) | Forecast pipeline, temporal safeguards, leakage prevention, and API flow |
| [docs/DATA_PIPELINE_DETAILED.md](docs/DATA_PIPELINE_DETAILED.md) | `dataset-creation.py` — ingestion, ERDDAP, buffers, Parquet output |
| [docs/dataset-creation-scientific-decisions.md](docs/dataset-creation-scientific-decisions.md) | Scientific rationale for data decisions |
| [docs/EVALUATION_AND_RESEARCH.md](docs/EVALUATION_AND_RESEARCH.md) | Paper and Hyak evaluation scripts (ablation, stability, metrics, spikes) |
| [docs/VISUALIZATIONS_GUIDE.md](docs/VISUALIZATIONS_GUIDE.md) | Chart and visualization interpretation |
| [docs/QUICK_START.md](docs/QUICK_START.md) | OS-specific local setup |
| [docs/HYAK_SETUP.md](docs/HYAK_SETUP.md) | Hyak (UW Klone) cluster workflow for cache pre-computation |
| [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) | Cache refresh on Hyak and Google Cloud Run deploy |

## Temporal Safeguards

The system enforces strict temporal integrity to prevent data leakage:

1. **Chronological splits** — training only uses `date <= anchor_date`
2. **Forward-only gap-filling** — exponential decay from past observations only (no future data in gap-filled targets)
3. **Satellite buffer** — 7-day processing delay
4. **Climate buffer** — 2-month reporting delay
5. **Observation-order lags** — past-only shifts on raw measurements
6. **Per-forecast categories** — DA risk levels computed from training data only
7. **Persistence features recomputed** — from real observations only, not gap-filled values
8. **Fresh model per test point** — no lookahead via shared state
9. **`_verify_no_data_leakage()`** — called for every prediction, raises `AssertionError` on violation

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

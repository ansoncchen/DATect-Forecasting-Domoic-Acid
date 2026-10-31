# DATect — Domoic Acid Forecasting System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React-blue.svg)](https://reactjs.org/)

## Overview

DATect forecasts domoic acid (DA), the harmful-algal-bloom toxin, in shellfish at 10 monitoring beaches on the Washington and Oregon coast, one week ahead. It combines satellite ocean data, climate indices, river discharge and past toxin measurements, with strict safeguards against using future data.

- 10 monitoring sites, 2003–2023
- Two-model ensemble (XGBoost + Random Forest) with per-site weights, feature subsets and prediction clipping; naive persistence is reported as a separate baseline
- Observation-order lag features for sparse, irregular measurements
- Leak-free evaluation: `_verify_no_data_leakage()` runs on every prediction, and models are scored only on real DA measurements
- Three-window chronological protocol: training before each anchor, validation 2019–2021, untouched holdout 2022–2023; a 5-seed bootstrap (seeds 42–46) confirms the holdout result
- All tuned values in one file, `config/tuned_hyperparameters.json`, with provenance for each key

## Performance

One-week-ahead forecasts on the untouched 2022–2023 holdout, scored against every
real DA measurement in the window (N = 409, no sampling):

| Metric | Value |
|---|---|
| R² [95% CI] | 0.497 [0.365, 0.601] (naive persistence 0.225) |
| MAE | 7.60 µg/g |
| Spike recall (actual ≥ 20 µg/g, alert at predicted ≥ 12) | 0.833 (F2 0.729) |
| Hybrid alert, recall on new spike onsets (seed-123 sample, 93 onsets) | 0.753, vs 0.129 for naive persistence |

All five Washington sites score R² 0.50–0.66 on the holdout. Newport (0.22) and Gold
Beach (below zero) in southern Oregon score lowest: their toxin levels have almost no
week-to-week persistence (lag-1 R² ≈ 0.05), so a one-week-ahead forecast has little to
work with.

All reported metrics, with their sources and history, are in [docs/CORRECTED_NUMBERS.md](docs/CORRECTED_NUMBERS.md).

## Quick Start

### Prerequisites

| Software | Version |
|----------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| Git | any |

```bash
# macOS
brew install python node git

# Ubuntu / Debian (NodeSource provides a current Node.js)
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
```

On Windows, install Python (tick "Add Python to PATH"), Node.js and Git from their websites.

### Run locally

```bash
git clone https://github.com/ansoncchen/DATect-Forecasting-Domoic-Acid.git
cd DATect-Forecasting-Domoic-Acid
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_datect.py
```

The dataset (`data/processed/final_output.parquet`) ships with the repo, so no data download is needed. `run_datect.py` installs the Python and frontend dependencies, starts the API and the dashboard, and opens the browser:

- Dashboard: http://localhost:3000
- API: http://localhost:8000 (health check: `curl http://localhost:8000/health`)
- API docs: http://localhost:8000/docs

It uses the fastest tools it finds: `uv` or `pip` for Python, `bun` or `npm` for the frontend, `granian` or `uvicorn` for the API. It stops anything already listening on ports 8000 and 3000 (via `lsof`, so not on Windows).

### Pre-computed cache (optional)

Retrospective views are slow to compute on a laptop. The cache is built on Hyak, never locally (see [docs/HYAK_SETUP.md](docs/HYAK_SETUP.md)):

```bash
# On a Hyak compute node, from the repo root:
python precompute_cache.py

# On your machine:
scp -r klone-login:/gscratch/stf/YOUR_UWNETID/DATect-Forecasting-Domoic-Acid/cache/ ./cache/
ENABLE_PRECOMPUTED_CACHE=true python run_datect.py
```

## Commands

| Command | What it does |
|---------|--------------|
| `python run_datect.py` | Start API + dashboard locally |
| `pip install -r requirements-dev.txt && pytest tests/` | Run the test suite |
| `cd frontend && npm run dev` / `npm run build` / `npm run lint` | Frontend development |
| `python precompute_cache.py` | Build the dashboard cache (Hyak only) |
| `python scripts/data/rebuild_final_output.py --recompute measurements` | Rebuild `da`/`pn` columns of the dataset offline after a change to the raw CSVs or their parsing |
| `python scripts/data/export_oad_features.py <checkpoint>` then `rebuild_final_output.py --recompute oad` | Switch or refresh the OAD input and features |
| `python dataset-creation.py` | Full dataset rebuild from source (re-downloads every satellite and climate series from NOAA ERDDAP; slow and often fails). Use the offline rebuild above for anything else |
| `bash scripts/eval/golden_baseline.sh <git-ref>` | Re-run every evaluation behind a reported number (Hyak) |
| `sbatch scripts/tune/tune_all.sbatch` | Re-tune `config/tuned_hyperparameters.json` end to end (Hyak) |
| `PROJECT_ID=<gcp-project> ./deploy/deploy_gcloud.sh` | Deploy to Google Cloud Run |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENABLE_PRECOMPUTED_CACHE` | `false` | `true` serves retrospective results from the pre-computed cache |
| `CACHE_DIR` | `./cache` | Location of the pre-computed cache |
| `REDIS_URL` | unset | Optional Redis cache backend (e.g. `redis://localhost:6379/0`) |
| `ALLOWED_ORIGINS` | unset | Comma-separated CORS origins; unset allows only localhost dev servers, so set it in production |
| `DATECT_PER_ANCHOR_TUNING` | `false` | `true` runs a per-forecast XGBoost grid search (1–3 min per forecast). Off by default, which matches how every reported number was produced |

Further `DATECT_*` variables for experiments (feature drops, per-site overrides, ablations) are documented in [CLAUDE.md](CLAUDE.md).

## Repository Layout

```
DATect-Forecasting-Domoic-Acid/
├── run_datect.py                   # Local launcher (API + dashboard + browser)
├── dataset-creation.py             # Full dataset build from source downloads
├── precompute_cache.py             # Dashboard cache pre-computation (Hyak)
├── config.py                       # Sites, data sources, structural constants
├── config/
│   └── tuned_hyperparameters.json  # All tuned values + provenance
├── forecasting/                    # Forecasting engine
│   ├── raw_forecast_engine.py      # Ensemble pipeline, leakage checks, retrospective evaluation
│   ├── raw_data_forecaster.py      # Raw DA loading, feature frames, leak-free test rows
│   ├── raw_data_processor.py       # Observation-order lag features
│   ├── per_site_models.py          # Per-site configuration (read from the tuned JSON)
│   ├── tuned_config.py             # Loader for config/tuned_hyperparameters.json
│   ├── measurements.py             # Raw DA / PN / date parsing (shared by data, eval and API)
│   ├── metrics.py                  # Spike detection and bootstrap metrics
│   ├── oad_features.py             # OAD ocean-anomaly features (16 per row, leak-safe)
│   ├── classification_adapter.py   # Risk categories and the spike classifier
│   ├── raw_model_factory.py        # Model builders (XGBoost, RF, classifier)
│   ├── ensemble_model_factory.py   # Model factory used by the API
│   ├── feature_utils.py            # Temporal features + transformers
│   ├── logging_config.py
│   └── validation.py               # Startup checks
├── backend/                        # FastAPI server (api.py, visualizations.py, cache_manager.py, redis_cache.py)
├── frontend/                       # React + Vite dashboard
├── data/
│   ├── raw/da-input/, raw/pn-input/  # Raw toxin and Pseudo-nitzschia measurements (one CSV per site)
│   └── processed/                  # final_output.parquet, oad_scores.parquet, oad_cloud_fractions.parquet
├── scripts/
│   ├── eval/                       # Evaluation, ablation, golden baseline
│   ├── tune/                       # tune_all.py re-tuning orchestrator
│   └── data/                       # Offline dataset rebuild, OAD feature export
├── results/                        # Committed outputs behind reported numbers
│                                   # (chronological, multi_seed, paper_metrics, final_verification,
│                                   #  holdout_validation, ablation, tuning_provenance)
├── oad/                            # Ocean Anomaly Detection subproject (MODIS autoencoder)
├── papers/                         # datect/, oad_mdpi/, oad_cvpr/, poster/
├── deploy/                         # cloudbuild.yaml, deploy_gcloud.sh
├── docs/                           # Documentation (below)
├── tests/                          # Pytest suite
├── Dockerfile.production
├── requirements.txt                # Runtime dependencies
├── requirements-dev.txt            # Tests, linting, figure scripts
└── CITATION.cff
```

`dataset-creation.py` caches satellite data in `data/intermediate/` (not committed).

## Using the Dashboard

1. Choose the forecast mode: a single forecast for one date and site, or a retrospective evaluation over historical points.
2. Pick the site, date and model: `ensemble` (recommended), `naive` (persistence) or `linear` (Ridge / logistic).
3. Run it. Single forecasts show the predicted DA, its interval and its risk category.

**Risk categories** (left-closed bins; a value on a boundary belongs to the higher category):

| Category | DA (µg/g) |
|----------|-----------|
| Low | 0 to < 5 |
| Moderate | 5 to < 20 |
| High | 20 to < 40 (at or above the 20 µg/g regulatory action level) |
| Extreme | 40 and above |

A spike is an actual DA of 20 µg/g or more; the regression alert fires when the forecast reaches 12 µg/g.

## Re-tuning and Re-evaluating

All tunable values live in `config/tuned_hyperparameters.json`. Re-tuning runs on Hyak; submit from the repo root:

```bash
sbatch scripts/tune/tune_all.sbatch
```

`tune_all.py` submits four sweeps (per-site grid search over ensemble weights and clipping with leak-free 3-fold chronological CV on pre-2022 data; per-site XGBoost perturbation check; global XGBoost sweep; spike-classifier sweep), waits for them, and writes a new JSON with provenance. `python scripts/tune/tune_all.py --assemble-only` rebuilds the JSON from existing sweep outputs. Individual stages and evaluations are listed in [docs/QUICKSTART_FOR_REVIEWERS.md](docs/QUICKSTART_FOR_REVIEWERS.md).

To re-run every evaluation behind a reported number for a commit, use the golden baseline (`scripts/eval/golden_baseline.sh`, then `scripts/eval/compare_golden.py` to diff two runs).

Env-var overlays (`DATECT_HPARAM_OVERRIDE_JSON`, `DATECT_SPIKE_CLASSIFIER_JSON`, etc.) apply on top of the JSON for experiments without editing it.

## Data Sources

- **Satellite** (NOAA CoastWatch ERDDAP): MODIS-Aqua chlorophyll-a, SST, PAR, fluorescence line height, K490 (8-day composites), plus chlorophyll and SST anomalies
- **Climate indices**: PDO, ONI, BEUTI (2-month reporting delay enforced)
- **Streamflow**: USGS Columbia River discharge (gauge 14246900)
- **Toxin and Pseudo-nitzschia data**: Washington and Oregon state monitoring programs
- **Ocean anomaly scores**: the `oad/` subproject's MODIS autoencoder (checkpoint `ae_3d_l32_c4_t4_s42_mae070`)

## Temporal Safeguards

1. **Chronological training**: each forecast trains only on rows with `date <= anchor_date` (anchor = target date − 7 days)
2. **Forward-only gap-filling**: training targets between real measurements decay from past observations only
3. **Satellite buffer**: 7-day processing delay
4. **Climate buffer**: 2-month reporting delay
5. **OAD buffer**: OAD features end 5 days before each weekly row, so the centered 8-day MODIS composites never reach the forecast window
6. **Observation-order lags**: past-only shifts over real measurements
7. **Persistence features**: recomputed from real observations only, not gap-filled values
8. **Fresh model per test point**: no shared state between forecasts
9. **`_verify_no_data_leakage()`**: raises `AssertionError` on every violation

## Documentation

| Document | Contents |
|----------|----------|
| [CLAUDE.md](CLAUDE.md) | Project conventions, environment variables, mechanical gotchas |
| [AGENTS.md](AGENTS.md) | Pointer for coding agents |
| [docs/CORRECTED_NUMBERS.md](docs/CORRECTED_NUMBERS.md) | All reported numbers and their history |
| [docs/QUICKSTART_FOR_REVIEWERS.md](docs/QUICKSTART_FOR_REVIEWERS.md) | Reproducing the paper's claims; evaluation and tuning script index |
| [docs/PIPELINE_DEEP_DIVE.md](docs/PIPELINE_DEEP_DIVE.md) | Forecast pipeline, temporal safeguards, API flow |
| [docs/DATA_PIPELINE_DETAILED.md](docs/DATA_PIPELINE_DETAILED.md) | `dataset-creation.py` |
| [docs/OAD_INTEGRATION_RESULTS.md](docs/OAD_INTEGRATION_RESULTS.md) | How OAD features enter DATect and what was learned |
| [docs/HYAK_SETUP.md](docs/HYAK_SETUP.md) | Hyak (UW Klone) cluster workflow |
| [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) | Cache refresh on Hyak and Cloud Run deploy |
| [docs/VISUALIZATIONS_GUIDE.md](docs/VISUALIZATIONS_GUIDE.md) | Reading the dashboard charts |
| [docs/superpowers/](docs/superpowers/) | Design spec and plan for the upcoming transport-aware forecasting work |
| [oad/README.md](oad/README.md), [oad/RESULTS.md](oad/RESULTS.md), [oad/OAD_STORYLINE.md](oad/OAD_STORYLINE.md) | OAD subproject: workflow, results, narrative |

## Google Cloud Deployment

Sync the Hyak-built cache into `./cache` first. The deploy script bakes it into the image; if `./cache` is missing or empty it runs `precompute_cache.py` locally, and if it exists it asks whether to re-generate it (answer N). Then:

```bash
gcloud auth login
export PROJECT_ID=your-gcp-project-id
export REGION=us-west1               # optional; this is the default
./deploy/deploy_gcloud.sh
```

The script enables the required APIs, builds with Cloud Build (`deploy/cloudbuild.yaml`) and deploys the `datect-api` Cloud Run service. See [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md).

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: requests` when starting | Activate the virtual environment and `pip install -r requirements.txt` |
| `python: command not found` | Use `python3` |
| Port 8000 or 3000 in use | The launcher frees them on macOS/Linux; otherwise `lsof -ti:8000,3000 \| xargs kill` |
| Node.js missing or older than 18 | Install a current version from [nodejs.org](https://nodejs.org/) or `brew upgrade node` |
| Retrospective runs are slow | Use the Hyak-built cache (`ENABLE_PRECOMPUTED_CACHE=true`) |
| Deploy fails on authentication | `gcloud auth list`, `gcloud auth application-default login`, check `PROJECT_ID` |

## License

Code is released under the [MIT License](LICENSE). If you use DATect in published research, please cite the manuscript (`papers/datect/datect_paper_mdpi.tex`, PDF alongside) and acknowledge the data providers listed above.

## Acknowledgments

- NOAA CoastWatch for satellite data (MODIS-Aqua via ERDDAP)
- USGS for Columbia River streamflow data
- Olympic Region HAB Partnership, WA DOH, OR DFW for toxin measurements

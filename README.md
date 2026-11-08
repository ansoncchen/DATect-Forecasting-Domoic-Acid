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
| R² [95% CI] | 0.516 [0.308, 0.654] (naive persistence 0.225) |
| MAE | 7.44 µg/g |
| Spike recall (actual ≥ 20 µg/g, alert at predicted ≥ 12) | 0.845 (F2 0.737) |
| Spike onsets caught by the hybrid alert (last known value < 20, measurement ≥ 20) | 19 of 27, alarming on 30% of quiet weeks |

The pooled R² leans on Coos Bay (0.80 there, 0.23 for the other sites together).
All five Washington sites score R² 0.50–0.67 on the holdout. Newport (0.13) and
Gold Beach (below zero) in southern Oregon score lowest: their toxin levels have
almost no week-to-week persistence (lag-1 R² ≈ 0.05), so a one-week-ahead forecast
has little to work with.

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

### Precomputed store (optional)

Forecasts and spectral analyses are slow to compute on a laptop. The precomputed store (Delta tables) is built on Hyak with Spark, never locally, and the deployed image serves it; `run_datect.py` points the API at `./store` when it exists (see [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)). A store's rows are served only where the config hash matches: the same code, data, model-library versions (`constraints.txt`), Python version (3.10) and CPU architecture (x86_64) as the Hyak env that built it. On a laptop (arm64, other versions) every row shows as stale on `/health` and requests are computed; check a store against the image instead (`docker build --platform linux/amd64 ...`, whose build fails unless the store is complete and current).

```bash
# On Hyak, at the commit you run (rows from other code, config or data are not served):
sbatch datect_spark/store.sbatch          # saves ~/datect-store/<commit>/

# On your machine:
rsync -a klone-login:datect-store/$(git rev-parse HEAD)/ ./store/
python run_datect.py
```

## Commands

| Command | What it does |
|---------|--------------|
| `python run_datect.py` | Start API + dashboard locally |
| `pip install -r requirements-dev.txt && pytest tests/` | Run the test suite |
| `cd frontend && npm run dev` / `npm run build` / `npm run lint` | Frontend development |
| `sbatch datect_spark/store.sbatch` | Build the precomputed store on Spark (Hyak only) |
| `python scripts/data/rebuild_final_output.py --recompute measurements` | Rebuild `da`/`pn` columns of the dataset offline after a change to the raw CSVs or their parsing |
| `python scripts/data/export_oad_features.py <checkpoint>` then `rebuild_final_output.py --recompute oad` | Switch or refresh the OAD input and features |
| `python dataset-creation.py` | Full dataset rebuild from source (re-downloads every satellite and climate series from NOAA ERDDAP; slow and often fails). Use the offline rebuild above for anything else |
| `bash scripts/eval/golden_baseline.sh <git-ref>` | Re-run every evaluation behind a reported number (Hyak) |
| `sbatch datect_spark/retune.sbatch` | Propose per-site ensemble weights (Spark, Hyak); adopting them is a separate change |
| `PROJECT_ID=<gcp-project> ./deploy/deploy_gcloud.sh` | Deploy to Google Cloud Run |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ALLOWED_ORIGINS` | unset | Comma-separated CORS origins; unset allows only localhost dev servers, so set it in production |
| `DATECT_PRECOMPUTED_DIR` | unset (`run_datect.py` sets `./store` if it exists) | Directory of the precomputed store's Delta tables (forecasts, retrospective, spectral, correlation; built on Spark). Rows whose config hash matches are served; everything else is computed on demand |
| `DATECT_PER_ANCHOR_TUNING` | `false` | `true` runs a per-forecast XGBoost grid search (1–3 min per forecast). Off by default, which matches how every reported number was produced |

Further `DATECT_*` variables for experiments (feature drops, per-site overrides, ablations) are documented in [CLAUDE.md](CLAUDE.md).

## Repository Layout

```
DATect-Forecasting-Domoic-Acid/
├── run_datect.py                   # Local launcher (API + dashboard + browser)
├── dataset-creation.py             # Full dataset build from source downloads
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
│   ├── ensemble_model_factory.py   # Model factory used by the engine
│   ├── feature_utils.py            # Temporal features + transformers
│   ├── logging_config.py
│   └── validation.py               # Startup checks
├── backend/                        # FastAPI server (api.py, visualizations.py, precomputed.py)
├── frontend/                       # React + Vite dashboard
├── data/
│   ├── raw/da-input/, raw/pn-input/  # Raw toxin and Pseudo-nitzschia measurements (one CSV per site)
│   └── processed/                  # final_output.parquet, oad_scores.parquet, oad_cloud_fractions.parquet
├── scripts/
│   ├── eval/                       # Evaluation, ablation, golden baseline
│   └── data/                       # Offline dataset rebuild, OAD feature export
├── results/                        # Committed outputs behind reported numbers
│                                   # (chronological, multi_seed, paper_metrics, final_verification,
│                                   #  holdout_validation, ablation, tuning_provenance)
├── datect_spark/                   # Spark + Delta Lake + MLflow on Hyak (re-tune, MODIS lake, serving)
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

1. Forecast: pick a site and a forecast date. The page shows the predicted DA with its error range, the spike alert level, the latest measurement before the forecast, and, for a past week, what happened.
2. Track record: how the forecasts performed on the 2022–2023 holdout and the 2019–2021 validation windows.
3. Data explorer and About: charts of the underlying data, and the method and its limits.

**Risk categories** (left-closed bins; a value on a boundary belongs to the higher category):

| Category | DA (µg/g) |
|----------|-----------|
| Low | 0 to < 5 |
| Moderate | 5 to < 20 |
| High | 20 to < 40 (at or above the 20 µg/g regulatory action level) |
| Extreme | 40 and above |

A spike is an actual DA of 20 µg/g or more; the regression alert fires when the forecast reaches 12 µg/g.

## Re-tuning and Re-evaluating

All tunable values live in `config/tuned_hyperparameters.json`. The per-site ensemble weights are the only values re-tuned, on Hyak, by `datect_spark/retune.py` (Spark; leak-free yearly folds 2015–2021; every run in MLflow). It proposes weights and never adopts them:

```bash
sbatch -o ~/datect-spark-logs/%x-%j.out -e ~/datect-spark-logs/%x-%j.err datect_spark/retune.sbatch
```

Evaluations are listed in [docs/QUICKSTART_FOR_REVIEWERS.md](docs/QUICKSTART_FOR_REVIEWERS.md); `results/tuning_provenance/` keeps the outputs of the earlier sweeps behind the current JSON.

To re-run every evaluation behind a reported number for a commit, use the golden baseline (`scripts/eval/golden_baseline.sh`, then `scripts/eval/compare_golden.py` to diff two runs).

Env-var overlays (`DATECT_HPARAM_OVERRIDE_JSON`, `DATECT_SPIKE_CLASSIFIER_JSON`, etc.) apply on top of the JSON for experiments without editing it.

## Running on Spark (Hyak)

`datect_spark/` runs open-source Spark 4.0 (local mode on one Hyak node), Delta Lake 4.0 and MLflow 3 on Hyak; it is not Databricks. The environment is a venv layered on the Hyak conda env (`/gscratch/scrubbed/ac283/envs/datect_spark`, `module load coenv/jdk/17.0.8`). The MODIS and serving jobs run from a `git archive` on the node's local disk (the shared scratch filesystems refuse writes intermittently when full) and copy their result JSONs to `~/datect-spark-results`; the MLflow sqlite database is on `/gscratch/stf`. Submit from any Hyak checkout of the branch, with logs outside the full filesystems:

```bash
L="-o $HOME/datect-spark-logs/%x-%j.out -e $HOME/datect-spark-logs/%x-%j.err"
sbatch $L datect_spark/modis_lake.sbatch                                  # MODIS NetCDFs -> bronze/silver/gold Delta
sbatch $L --export=ALL,STEPS=reference datect_spark/modis_lake.sbatch     # time the reference cube scripts
sbatch $L --export=ALL,MAIN_SHA=$(git rev-parse origin/main) datect_spark/serving.sbatch  # latency, precompute, parity
sbatch datect_spark/retune.sbatch                                         # weights x clip re-tune (from a worktree)
```

Every job checks its output against the existing pipeline exactly. Measurements, parity checks and caveats are in [datect_spark/RESULTS.md](datect_spark/RESULTS.md).

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
| [datect_spark/RESULTS.md](datect_spark/RESULTS.md) | Spark, Delta Lake and MLflow on Hyak: re-tune, MODIS lake, forecast serving; parity, sizes, timings |
| [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) | Cache refresh on Hyak and Cloud Run deploy |
| [docs/VISUALIZATIONS_GUIDE.md](docs/VISUALIZATIONS_GUIDE.md) | Reading the dashboard charts |
| [docs/superpowers/](docs/superpowers/) | Design spec and plan for the upcoming transport-aware forecasting work |
| [oad/README.md](oad/README.md), [oad/RESULTS.md](oad/RESULTS.md), [oad/OAD_STORYLINE.md](oad/OAD_STORYLINE.md) | OAD subproject: workflow, results, narrative |

## Google Cloud Deployment

Copy the Hyak-built store into `./store` first, built at the commit you deploy. The deploy script bakes it into the image and stops if `./store` is missing or empty. Then:

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
| Forecasts are slow | Use the Hyak-built store in `./store`; `/health` shows whether its rows match this code (stale rows are computed on demand) |
| Deploy fails on authentication | `gcloud auth list`, `gcloud auth application-default login`, check `PROJECT_ID` |

## License

Code is released under the [MIT License](LICENSE). If you use DATect in published research, please cite the manuscript (`papers/datect/datect_paper_mdpi.tex`, PDF alongside) and acknowledge the data providers listed above.

## Acknowledgments

- NOAA CoastWatch for satellite data (MODIS-Aqua via ERDDAP)
- USGS for Columbia River streamflow data
- Olympic Region HAB Partnership, WA DOH, OR DFW for toxin measurements

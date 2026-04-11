# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Commands

```bash
# Run the complete system (starts backend API + frontend + opens browser)
python3 run_datect.py
# After starting: Backend → http://localhost:8000 | Frontend → http://localhost:3000 | API docs → http://localhost:8000/docs
# Picks uv vs pip, bun vs npm, granian vs uvicorn from PATH; installs from requirements.txt

# Pre-compute cache (MUST run on Hyak, not locally)
python3 precompute_cache.py

# Regenerate dataset (30-60 min process, only when data changes)
python3 dataset-creation.py

# Deploy to Google Cloud Platform
./deploy_gcloud.sh

# Frontend development commands (from frontend/ directory)
cd frontend && npm run dev      # Development server
cd frontend && npm run build    # Production build
cd frontend && npm run lint     # ESLint validation
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENABLE_PRECOMPUTED_CACHE` | `false` | Set `"true"` to use pre-computed cache locally |
| `CACHE_DIR` | `./cache` | Path to pre-computed cache directory |
| `REDIS_URL` | unset | Redis connection URL for 100x faster cache reads (e.g. `redis://localhost:6379/0`) |
| `ALLOWED_ORIGINS` | unset | Comma-separated browser origins for CORS (e.g. `https://app.example.com`). If unset, defaults to `http://localhost:3000` and `http://localhost:5173` only — **set this in production** so the API is not limited to localhost. |

## Environment Notes

- **`POST /api/config`** only updates the running process’s in-memory `config` values (it does **not** edit `config.py` on disk). Restart the API to reload from `config.py`; for durable deploy settings use env vars or your orchestration layer.
- **Do not run heavy scripts locally** (precompute_cache.py). These must be run on the Hyak compute cluster.
- Local development is for code editing, review, lightweight testing, and running the dashboard with pre-computed cache.
- See `docs/HYAK_SETUP.md` for cluster workflow.

## System Architecture

DATect is a machine learning system for forecasting harmful algal bloom toxin concentrations (domoic acid) along the Pacific Coast. It uses a **two-model ML ensemble** (XGBoost + Random Forest), with naïve persistence as an external standalone baseline, with per-site hyperparameter tuning and leak-free validation on raw DA measurements.

### Core Components

**Forecasting Engine** (`forecasting/`)

| File | Purpose |
|------|---------|
| `raw_forecast_engine.py` | **Main engine** — ensemble pipeline with per-site tuning |
| `raw_data_forecaster.py` | Raw DA loading, feature frame building, leak-free test rows |
| `raw_data_processor.py` | Observation-order lag features (not grid-shift) |
| `per_site_models.py` | Per-site hyperparams, feature subsets, ensemble weights (10 sites) |
| `raw_model_factory.py` | Standalone model builders (XGB, RF, classifier) |
| `ensemble_model_factory.py` | Class-based wrapper matching API's ModelFactory interface |
| `classification_adapter.py` | Threshold + dedicated XGBoost classifier for 4 DA categories |
| `feature_utils.py` | Shared temporal features + transformer creation |
| `validation.py` | System startup validation |

**Web Interface**
- `backend/api.py`: FastAPI server providing forecasting endpoints
- `frontend/`: React + Vite interface for dashboards and visualizations
- `backend/visualizations.py`: Chart generation (correlation matrices, time series)
- `backend/cache_manager.py`: Pre-computed cache access (file-based + Redis backend)
- `backend/redis_cache.py`: Optional Redis caching (100x faster; set `REDIS_URL` to enable)

**Data Pipeline**
- `dataset-creation.py`: Downloads and processes 21 years of satellite/environmental data
- `config.py`: Centralized configuration for data sources, model parameters, temporal settings

**Technical Documentation** (`docs/`)
`PIPELINE_DEEP_DIVE.md` (forecasting + safeguards), `DATA_PIPELINE_DETAILED.md` (`dataset-creation.py`), `dataset-creation-scientific-decisions.md`, `EVALUATION_AND_RESEARCH.md` (paper/Hyak scripts), `VISUALIZATIONS_GUIDE.md`, `QUICK_START.md`, `HYAK_SETUP.md`, `DEPLOYMENT_GUIDE.md`.

### Key Design Principles

**Interpolated-Training Ensemble Forecasting**: Uses a two-model ML ensemble (XGBoost + Random Forest) with per-site weighted blending. Naïve persistence is computed separately as an external standalone baseline. Trains on all rows (real + gap-filled DA, ~5x more data); tests on raw DA measurements only. Controlled by `USE_INTERPOLATED_TRAINING` in `config.py`.

**Temporal Integrity**: Environmental features come from anchor date (test_date - 7 days). Persistence features recomputed from training data only. No future data leakage.

**Per-Site Customization**: Each of the 10 Pacific Coast sites has custom XGBoost/RF hyperparameters, feature subsets, ensemble weights, and prediction clipping via `per_site_models.py`. These were hand-tuned on the seed=42 dev set; `scripts/eval/paper_stability_study.py` validates their robustness across seeds and perturbations.

**Observation-Order Lag Features**: Instead of grid-shift lags, uses the Nth most recent actual observation, which is critical for sparse/irregular measurement data.

**DA Risk Categories**: Low (0-5), Moderate (5-20), High (20-40), Extreme (40+ µg/g). Both threshold-based and ML-based classification supported.

**Model options**: `ensemble` (XGB + RF — two-model ML blend), `naive` (persistence: most recent DA at/before anchor — standalone external baseline), `linear` (Ridge full-feature regression / Logistic classification). Linear is a competitor, not a baseline.

**Streamlined Feature Pipeline**: After systematic ablation, the pipeline uses only features with confirmed impact: persistence (last DA, weeks since spike), observation-order lags (4 values + 2 recency + 1 trend), rolling stats (mean/max at 4-week; std/max at 8/12-week), 3 temporal encodings (sin/cos day-of-year, month), environmental (SST, BEUTI, PDO, ONI, discharge, FLH, SST-anom), and pn_log. Six parquet columns (lat, lon, modis-par, modis-k490, chla-anom, modis-chla) are dropped before training.

**Configuration Management**: `config.py` contains all system parameters:
- 10 Pacific Coast monitoring sites with coordinates
- Satellite data URLs (MODIS ocean color, SST, chlorophyll)
- Model hyperparameters (XGBoost, RF, Ridge, classification)
- Per-site configuration via `per_site_models.py`
- Raw pipeline params: ZERO_IMPORTANCE_FEATURES, PREDICTION_CLIP_Q, CALIBRATION_FRACTION, PARAM_GRID

## Hyak Workflow

**precompute_cache.py** (deployment cache):
1. Run on Hyak: `python precompute_cache.py`
2. SCP cache: `scp -r klone-node:/.../cache/ ./cache/`
3. Run locally: `python run_datect.py`

**Paper evaluation scripts** (`scripts/eval/` — run from repo root):
1. Stability: `python3 scripts/eval/paper_stability_study.py` (~1.5 hrs at 20% sample; full run can be ~3 hrs)
2. Smoke test: `python3 scripts/eval/paper_stability_study.py --quick` (1% sample)
3. Tables from stability JSON: `python3 scripts/eval/paper_stability_table.py` or `--latex`
4. Phase 1B only: `python3 scripts/eval/paper_stability_study.py --phase 1b`
5. Paper metrics / CIs: `python3 scripts/eval/eval_paper_metrics.py`
6. Full Hyak sweep: `bash run_full_validation.sh`

## Development Workflow

1. **Data Changes**: Run `python dataset-creation.py` to regenerate the processed dataset
2. **Model Changes**: Modify parameters in `config.py` or `per_site_models.py`, then test on Hyak
3. **Frontend Changes**: Use `cd frontend && npm run dev` for development server
4. **Full System Testing**: Use `python run_datect.py` to test complete integration with pre-computed cache

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Ensemble R² | 0.215 (independent test, seed=123) / 0.414 (dev, seed=42) / 0.315 (temporal holdout, 2019+) | Maximize |
| Ensemble MAE | 6.42 µg/g | Minimize |
| Spike recall | 0.558 ensemble / 0.859 hybrid alert | Maximize recall |
| Transition recall | 0.734 hybrid alert / 0.236 naïve persistence | Maximize recall |

## No Data Leakage Guarantees

- Training only uses `date ≤ anchor_date` (includes gap-filled rows for more data)
- `da_raw` and `da` dropped from test features
- Lag features use observation-order past-only shifts
- Persistence features and Naive baseline recomputed from **real observations only** (not gap-filled)
- Test/evaluation uses only real raw DA measurements
- Fresh model per test point (no lookahead)
- `verify_no_data_leakage()` called for every prediction

**Temporal holdout caveat**: The temporal holdout (2019+) evaluates post-2019 data that was never used as training targets, but per-site tuning decisions (ensemble weights, feature subsets, hyperparameters in `per_site_models.py`) were made on the seed=42 dev set which includes post-2019 test points. This means tuning decisions were indirectly informed by post-2019 patterns. `scripts/eval/paper_stability_study.py` validates that these choices are robust across seeds and perturbations.

## Stability Study Results (Phase 1)

Multi-seed (5 seeds) and perturbation (13 experiments) validation confirms current design choices:

| Decision | Sensitivity | Conclusion |
|----------|-------------|------------|
| RF hyperparameters | |ΔR²| < 0.001 across 4 configs | RF genuinely robust — no tuning needed |
| Per-site feature subsets | |ΔR²| < 0.006 | Persistence dominates; subsets barely matter pooled |
| Per-site clipping thresholds | ΔR² = −0.030 when relaxed | Current 0.95–0.98 quantiles are well-calibrated |
| Winner-take-all model selection | ΔR² = −0.028 (RF→XGB), −0.005 (XGB→RF) | Current assignments near-optimal |
| Monotonic constraints | ΔR² = +0.0003 | Negligible effect |
| Per-site config overall | ΔR² = −0.103 without it | Biggest lever — validates per-site customization |

**Regional stability**: WA sites are stable across seeds (mean R²≈0.61, std≈0.05). OR sites are near-random (mean R²≈−0.03, std≈0.25) — this is a data scarcity/oceanographic issue, not a tuning problem.

**Go/no-go**: No perturbation exceeds the noise floor → Phase 2 (Optuna/grid search) skipped. Current config validated.

## Autocorrelation Ceilings

The ρ² ceiling in `autocorrelation_diagnostic.py` bounds a **persistence-only** forecast. Models using environmental features (SST, BEUTI, PDO, discharge, etc.) can exceed this bound. OR sites above ρ² are exploiting environmental signal, not overfitting.

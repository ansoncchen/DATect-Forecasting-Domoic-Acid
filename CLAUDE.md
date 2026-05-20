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

## Mechanical Gotchas

- **Local Python**: use `.venv/bin/python` (pandas, optuna, xgboost installed there; system `python3` does not have project deps).
- **Hyak Python**: `/gscratch/stf/ac283/envs/datect_scratch/bin/python` (conda env with torch + xarray + zarr).
- **Hyak Slurm**: prefer `--partition=ckpt --account=stf-ckpt --requeue` for batch eval/tuning. The `compute` partition often hits `QOSGrpMemLimit` from group memory pressure even at modest requests (32G+). ckpt is preemptible but free; use sqlite-backed studies for resumability.
- **Hyak log files**: under sbatch, Python `print()` is line-buffered to `.out` (often appears empty for long stretches); `tqdm` writes to `.err`. To check ablation/tuning progress, `tail -1 logs/*.err | tr "\r" "\n" | tail -3`.
- **Adding columns in `dataset-creation.py`**: the `final_core_cols` block (~line 1206) is a hardcoded allow-list — any new column **silently dropped** unless added there. Pattern: insert the join after `add_satellite_data()` (~line 1200), then append `extra_cols = [c for c in NEW_COLS if c in final_data.columns]` to `final_cols`. Note: the streamflow merge at line 955 is inside a helper function, not the main flow.
- **`final_output.parquet` date column** is stored as object string `"MM/DD/YYYY"`, NOT a Timestamp. Always `pd.to_datetime(df["date"])` before joins/comparisons; convert back via `.dt.strftime("%m/%d/%Y")` before writing.
- **Single forecast**: `engine.generate_single_forecast(data_path, forecast_date, site, task, model_type)` — all 5 positional. Heavy locally (per-anchor XGB tuning takes 1-3 min per forecast); for end-to-end smoke tests use the retrospective eval on a small subset instead.
- **Env-var ablation hooks in `config.py` / `per_site_models.py`**: `DATECT_EXTRA_DROP_FEATURES` (CSV append to `ZERO_IMPORTANCE_FEATURES`), `DATECT_USE_INTERPOLATED_TRAINING`, `DATECT_USE_PER_SITE_MODELS`, `DATECT_LAG_FEATURES`, `DATECT_USE_MONOTONIC_CONSTRAINTS`, `DATECT_CLIP_Q_OVERRIDE`, `DATECT_RF_PARAMS_JSON`, `DATECT_FEATURE_SUBSET_MODE`, `DATECT_HPARAM_OVERRIDE_JSON`, `DATECT_SPIKE_CLASSIFIER_JSON`, `DATECT_OAD_ON_SMALL_N`. Prefer these over new flags — `paper_ablation_study.py` already runs subprocess + env-var pattern.
- **Hyperparameter tuning protocol** (added in `oad-integration` branch): use the **3-window chronological split** — training = pre-anchor data (engine-enforced); validation = retrospective points in `[2019-01-01, 2022-01-01)` via `TUNE_VAL_START`/`TUNE_VAL_END` env vars (Optuna objective); holdout = `[2022-01-01, 2024-12-31]` (final unbiased report only). Random-seed sampling (existing `paper_ablation_study.py`) is fine for **model-vs-model comparison** but NOT for hyperparameter selection — it has temporal autocorrelation between sampled "train" and "test" anchors. Use `scripts/eval/validate_tuned_on_holdout.py --compare baseline tuned` for the REAL / OVERFITTING / NEUTRAL / NULL verdict.

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

## Ocean Anomaly Detection (OAD) subproject

A parallel subproject lives at **`ocean anomaly detection/`** (branch `ocean-anomaly-v2`). It is an **unsupervised convolutional autoencoder over 4-channel MODIS Aqua imagery** (chla, Kd490, nflh, SST) that produces a per-region scalar "ocean state anomaly score" for the U.S. Pacific Northwest coast.

- **Trained on**: 22-year cube (2003–2024) at stride-2 / 0.025° resolution, ~4,700 daily rolling 8-day composites × 4 channels × 321 × 409.
- **Headline checkpoint**: `ae_3d_l32_c4_t4_s42_mae070` — 3D ConvAE3D with Phase C masked-autoencoder training (70% random pixel hiding). **Strongest defensive finding**: at lead=7 days (the informative-horizon test that the 1-day-ahead headline number does not pass cleanly), AE_3d_mae070 is the **only method** with positive R² in every PNW region (0.10–0.26), while climatology baselines B1/B2 collapse to ≤ 0 (B2 goes *negative* — anti-predicts) and PCA was already at 0. The 1-day-ahead R²=0.87 in SW Washington is real but inflated by 8-day composite overlap. Moderate cloud-cover confound: Pearson r ≈ +0.4–0.5 with valid-pixel fraction (~24% variance). For DATect integration, **use this same mae070 checkpoint** — the cleaner-cloud variant mae050 collapses faster at long leads (R²=−0.17 vs +0.19 at lead=7 in Olympic Coast), so the cloud tradeoff isn't worth the multi-step penalty.
- **5 regions** (1 envelope + 4 alongshore bands) — each of DATect's 10 sites maps to exactly one region (mapping documented in `ocean anomaly detection/RESULTS.md`).
- **Score parquets**: `ocean anomaly detection/outputs/scores/*.parquet`, columns `date, region, method, aggregation, score`.

### Integration into the main DATect forecast (planned)

The OAD score is a candidate new feature column for `forecasting/raw_data_processor.py`. **Three integration constraints** (all from `RESULTS.md` §SANITY-CHECK CAVEATS):

1. **Centered-composite leakage**: MODIS 8-day composites are **CENTERED** on the labeled date (`long_name="Centered Time"`), so a score at date *t* contains 3–4 days of "future" data. The leakage-safe lag is `test_date − 12` (i.e. anchor − 5), which puts the score's composite window entirely before DATect's `anchor = test_date − 7` convention.
2. **Use the lead=7+ regime**: at the trivial 1-day-ahead lead the AE's R² is inflated by 8-day composite overlap (0.87). The real, usable signal is at lead ≥ 7 days, where R² is 0.10–0.26 — modest but PCA collapses to 0 there too. The leakage lag in constraint (1) already puts us in this regime.
3. **Cloud-fraction confound**: OAD score has Pearson r ≈ +0.4–0.5 with in-region valid-pixel fraction. Add `oad_valid_pixel_fraction` as a parallel feature so XGBoost can learn to discount cloudy weeks (preferred), or pre-regress cloud fraction out of the score before merging.

Suggested 8 new features per (site, date) row: `oad_score`, `oad_score_lag1week`, `oad_score_lag2week`, `oad_score_30day_mean`, `oad_score_30day_max`, `oad_score_30day_trend`, `oad_score_zscore_doy`, **`oad_valid_pixel_fraction`** (cloud control). Evaluate via `scripts/eval/eval_paper_metrics.py` baseline vs +OAD on the same retrospective rows. Source parquet: `ocean anomaly detection/outputs/scores/ae_3d_l32_c4_t4_s42_mae070.parquet` (best multi-step skill, only method with positive R² at lead=7 in every region).

See `ocean anomaly detection/RESULTS.md` and `ocean anomaly detection/IMPLEMENTATION_PLAN.md` for full design + validated numbers.

## Quick reference: subproject docs

- `ocean anomaly detection/README.md` — Hyak-first workflow
- `ocean anomaly detection/IMPLEMENTATION_PLAN.md` — design + completion status of Phases A/B/C
- `ocean anomaly detection/RESULTS.md` — validated numbers (per-region E4 forecastability tables, MAE ratio comparison, annual cycle plots)
- `ocean anomaly detection/AGENTS.md` — workspace facts for agents continuing OAD work

## Agent / development guidance

These are conventions that have accumulated from prior conversations. They are
project-wide rules, not optional preferences — follow them by default unless the
user explicitly opts out.

### Process rules

1. **Keep gap-fill (synthetic training targets) separate from forecast model choice.**
   They are two independent design dimensions (what data fills sparse DA between
   real samples vs which model — XGB, RF, MLP — predicts on that data). Mixing
   them up in experiments leads to incorrect attribution of improvements.
2. **Leak-free, past-only training rules are non-negotiable.** Any alternative
   to causal exponential decay gap-fill (e.g. bidirectional imputers) needs an
   explicit story for why it does not leak future information into training rows.
   "It looks like it works on the panel" is not enough.
3. **Judge improvements on the leak-free raw retrospective**, not on dense ISO-week
   panel imputation quality. The weekly panel has gap-filled `da`; only the raw
   shore measurements are the actual prediction targets the system should be
   evaluated against.
4. **Prefer simple plain-English explanations** when the conversation moves
   between synthetic-data design and prediction-model ablations. Those two
   threads are easy to conflate.
5. **Run all heavy compute on Hyak** (`/gscratch/stf/ac283/...`). Local laptop
   is for editing code and reviewing figures. Cluster workflow in `docs/HYAK_SETUP.md`.
6. **Memory hygiene**: before pasting a file path from a learned-fact note into
   a prompt or instruction, verify it exists with `ls`. Past agents have
   hallucinated filenames like `quick_raw_retrospective_compare.py`; the real
   entry points are listed in "Hyak Workflow" above.

### Useful workspace facts

- Raw DA volume is in the **thousands** of shore rows and **thousands** of
  site-weeks with real measurements. The weekly panel is dense with gap-filled
  `da`, but evaluation for forecasting skill should always track raw measurements.
- Quick retrospective comparisons and small **MLP / sklearn-style baselines**
  at modest sample fractions are **CPU-viable on Hyak**; a GPU is not required
  for that evaluation tier (reserve GPU partitions for the autoencoder training
  in the OAD subproject).
- SSH ControlMaster for Hyak is set up as `klone-login`. It expires periodically
  (2FA timeout); user re-auths from their terminal with `ssh klone-login 'whoami'`
  when needed.

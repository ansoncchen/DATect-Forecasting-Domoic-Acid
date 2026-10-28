# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository. Model
results live in `docs/CORRECTED_NUMBERS.md`; this file holds how to work here.

## Quick commands

```bash
python3 run_datect.py                         # API (:8000) + frontend (:3000) + browser
.venv/bin/python -m pytest tests/ -q          # test suite (~1 min, order-independent)
cd frontend && npm run lint && npm run build  # frontend checks

# Data: rebuild final_output.parquet offline (never re-download; see below)
python scripts/data/rebuild_final_output.py --recompute measurements   # da, pn
python scripts/data/rebuild_final_output.py --recompute oad            # oad_* features
python scripts/data/export_oad_features.py ae_3d_l32_c4_t4_s42_mae070  # OAD input

# Hyak: every evaluation that backs a reported number, in a disposable worktree
bash scripts/eval/golden_baseline.sh <git-ref> <label>
python scripts/eval/compare_golden.py <golden_root>/<a> <golden_root>/<b>

PROJECT_ID=<gcp-project> ./deploy/deploy_gcloud.sh                     # Cloud Run
```

Local Python is `.venv/bin/python` (system `python3` lacks the deps). Install with
`pip install -r requirements.txt -r requirements-dev.txt`.

## Hard rules

1. **Never re-download ERDDAP data.** MODIS satellite chunks, PDO, ONI and BEUTI all come
   from NOAA ERDDAP, which is too unreliable to re-fetch, and there is no local satellite
   cache, so `dataset-creation.py` end-to-end would re-download everything. Treat the
   environmental columns in `data/processed/final_output.parquet` as frozen and use
   `scripts/data/rebuild_final_output.py` for anything derived from local files.
2. **Heavy compute runs on Hyak** (`precompute_cache.py`, retrospective/multi-seed/ablation
   evals). The laptop is for editing, tests and the dashboard.
3. **Leak-free, past-only training is non-negotiable.** Training uses `date <= anchor_date`
   (`anchor = test_date - 7 days`); persistence features and the naive baseline come from
   real observations only; test points are raw DA measurements only. Any new feature or
   imputer needs an explicit story for why it cannot see the future.
4. **Behavior-neutral refactors must reproduce the golden baseline exactly**
   (`compare_golden.py`: prediction parquets byte-identical; metric tables may differ only
   by ~1e-12 rounding). Behavior-changing fixes go in their own commit with before/after
   numbers, then the reported numbers are re-run and updated.
5. **Judge improvements on the leak-free raw retrospective**, not on the gap-filled weekly
   panel. Keep the gap-fill (synthetic training targets) and the model choice as separate
   experimental dimensions.

## Conventions that are easy to get wrong

- **Spike rule:** a value at a threshold counts. Spike = actual DA `>= 20` ug/g; the
  regression alert fires at predicted `>= 12`. Categories are left-closed bins
  `[0,5) [5,20) [20,40) [40,inf)` (`pd.cut(..., right=False)`). Use
  `forecasting/metrics.py` (`exceeds`, `detection_scores`, `bootstrap_samples`) instead of
  writing comparisons or F-scores by hand.
- **Measurement parsing** is in `forecasting/measurements.py`: `"<1"` (below detection) is
  LOD/2 = 0.5; PN counts with thousands separators (`"12,345"`) are parsed; dates are
  stripped of whitespace. All three readers (dataset build, evaluation targets, API) use it.
- **Same-day DA samples:** ~1,200 site-days have several samples that often disagree
  (Gold Beach 2019-01-24 = [0, 190]; the CSVs record no species or station).
  `load_raw_da_measurements` returns one row per site-day at the **max**, like the weekly
  MAX used everywhere else. Never pick "the first" duplicate: it made the evaluation
  target depend on sort order and inflated the published holdout R² (0.485 vs 0.419).
- **Seeds:** `config.RANDOM_SEED` only chooses retrospective test points;
  `config.MODEL_SEED` (42) seeds every model. Joblib/Spark workers re-import `config`, so
  patching `config` in the parent never reaches them; vary behavior through env vars.
- **Per-anchor XGB tuning is off by default** (`DATECT_PER_ANCHOR_TUNING=true` to enable). It
  changes ~23% of predictions without improving skill and costs 1-3 min per forecast; every
  reported number was produced with it off.
- **Single forecasts use the evaluated models:** `generate_single_forecast` calls
  `_run_single_raw_validation` (the retrospective core) and only adds intervals and
  categories. Intervals use XGBoost `reg:quantileerror` (not `reg:quantile`, which does not
  exist) plus the spread of the Random Forest's trees.
- **The pooled spike classifier** is trained on a pooled imputer/MinMax encoding of all
  sites; test rows must be encoded with that same transformer (returned by
  `_build_pooled_spike_data`), not the per-site one.
- **`SITE_SPECIFIC_CONFIGS[site]['ensemble_weights']` is a tuple `(w_xgb, w_rf, w_naive)`**,
  not a dict. Weights, clip quantiles, feature subsets and hyperparameters come from
  `config/tuned_hyperparameters.json` (grid-search winners), loaded by
  `forecasting/tuned_config.py`.
- **`final_output.parquet`:** lowercase `date`/`site` columns; `date` is a `"MM/DD/YYYY"`
  string (`pd.to_datetime` before joins, `.dt.strftime("%m/%d/%Y")` before writing); 33
  columns including 16 OAD features.
- **Adding a column in `dataset-creation.py`:** the `final_core_cols` list in `main()` is an
  allow-list; a new column is silently dropped unless added there (or appended like the
  `sat_*` / OAD columns).
- **The decay fill cascades:** each filled week seeds the next gap, so `*_MAX_GAP_WEEKS` does
  not limit fills; only weeks before a site's first measurement become 0.
- **API handlers are plain `def`** (FastAPI runs them in its thread pool). Model work inside
  `async def` blocks the event loop for every user. Settings such as task/model travel with
  each request; the server holds no per-user state (there is no `POST /api/config`).
- **Tests:** use the `fast_models` (class-scoped model shrinking, restored afterwards) and
  `fresh_import` (re-import config modules for one test, originals restored) fixtures in
  `tests/conftest.py`. Never mutate shared config dicts without restoring them; tests must
  pass in any file order.

## Env-var hooks (ablations and experiments)

Read at import by `config.py` / `forecasting/per_site_models.py`, so set them in the
subprocess environment (see `scripts/eval/_subprocess.py`):

| Variable | Effect |
|---|---|
| `DATECT_PER_ANCHOR_TUNING` | `true` enables per-anchor XGB grid tuning (default off) |
| `DATECT_USE_PER_SITE_MODELS` | `false` disables per-site configs (equal XGB/RF weights) |
| `DATECT_USE_INTERPOLATED_TRAINING` | `false` trains on real DA rows only |
| `DATECT_USE_MONOTONIC_CONSTRAINTS` | `false` drops XGBoost monotone constraints |
| `DATECT_LAG_FEATURES` | lag list, e.g. `1,2`; `none` disables lag features |
| `DATECT_EXTRA_DROP_FEATURES` | CSV of extra features to drop |
| `DATECT_CLIP_Q_OVERRIDE` | `none` disables clipping; a number replaces every clip quantile |
| `DATECT_RF_PARAMS_JSON` / `DATECT_SPIKE_CLASSIFIER_JSON` | overlay RF / spike-classifier params |
| `DATECT_HPARAM_OVERRIDE_JSON` | per-site overrides, incl. `feature_subset` |
| `DATECT_FEATURE_SUBSET_MODE` | `all` removes per-site subsets; `minimal` forces a 4-feature subset |
| `DATECT_TEMPORAL_HOLDOUT_CUTOFF` | holdout start for `eval_paper_metrics --temporal-holdout` (default 2022-01-01) |
| `DATECT_PARALLEL_BACKEND` | joblib backend (default `loky`) |

Backend/deploy variables (`ENABLE_PRECOMPUTED_CACHE`, `CACHE_DIR`, `REDIS_URL`,
`ALLOWED_ORIGINS`, `PORT`, `WORKERS`, `DATECT_USE_GRANIAN`, `NODE_ENV`, `VITE_API_URL`) are
documented in `docs/DEPLOYMENT_GUIDE.md`. Set `ALLOWED_ORIGINS` in production.

## Evaluation protocol

- **Three chronological windows:** training = pre-anchor data (engine-enforced); validation =
  `[2019-01-01, 2022-01-01)` (tuning objective); holdout = `[2022-01-01, 2024-01-01)`
  (final report only, never used for selection). Defined in the tuned JSON's `eval_windows`.
- **Headline = deterministic chronological eval** (`scripts/eval/chronological_eval.py`):
  every real DA measurement in the window, no sampling. Confirm with the 5-seed bootstrap
  (`multi_seed_baseline.py`, seeds 42-46 at 20%). Single-seed R² at N~1,200 is noise-dominated
  (std across seeds ~0.08 pooled, ~0.15 on the holdout); never report one seed alone.
- **Never tune all dimensions at once.** A full Optuna search over 18 dims overfit badly
  (validation +0.07 R², holdout -0.16). Tune only ensemble weights and clip parameters with
  constrained chronological CV (`grid_search_weights_clip.py`), and verify on the holdout
  (`validate_tuned_on_holdout.py`). Per-site fold CV with < 50 test points per fold is
  unreliable.
- **Results files:** `results/chronological` (headline), `results/multi_seed`,
  `results/paper_metrics`, `results/ablation`, `results/final_verification`,
  `results/holdout_validation`, `results/tuning_provenance`. The dashboard reads the first
  two through `GET /api/metrics/summary`.

## Hyak

- SSH alias `klone-login` (ControlMaster; re-auth with `ssh klone-login 'whoami'` when it
  expires). Conda env `/gscratch/stf/ac283/envs/datect_scratch/bin/python` (Python 3.10).
- Submit sbatch files from the repo root: they `cd "$SLURM_SUBMIT_DIR"` and accept
  `DATECT_PY`. Prefer `--partition=ckpt --account=stf-ckpt --requeue`; exclusive nodes can
  wait hours on a busy queue, and results do not depend on core count. `logs/` must exist
  (it is tracked) because Slurm opens `--output=logs/...` before the job runs.
- Run long jobs in a disposable worktree (`golden_baseline.sh <ref> <label> [sbatch]`) so
  relative outputs never dirty a checkout. Parallel jobs must not write the same file (four
  chains writing `final_output.parquet` once corrupted it).
- **Do not poll the login node in loops.** A fail2ban-style block hits your IP after
  ~50-100 failed SSH attempts per hour (symptom: `Connection refused` everywhere while jobs
  keep running); wait 15-30 min, don't reboot. When waiting remotely for a process, don't
  use `pgrep -f <pattern>` from a shell whose own command line contains the pattern.
- `git fetch` over HTTPS on Hyak can hang on a credential prompt; use
  `GIT_TERMINAL_PROMPT=0`. The old checkout at `/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid`
  holds untracked OAD data; work happens in worktrees such as `/gscratch/stf/ac283/datect-cleanup`.
- OAD data lives behind `/gscratch/stf/ac283/oad_current` (symlink to that checkout's
  `ocean anomaly detection/`); OAD sbatch files default `OAD_ROOT` to it.

## Architecture

| Path | Purpose |
|---|---|
| `forecasting/raw_forecast_engine.py` | Engine: retrospective evaluation, single forecasts, intervals, spike classifier |
| `forecasting/raw_data_forecaster.py`, `raw_data_processor.py` | Raw DA loading, feature frame, observation-order lags, leak-free test rows |
| `forecasting/per_site_models.py`, `tuned_config.py` | Per-site configs from `config/tuned_hyperparameters.json` + env overlays |
| `forecasting/metrics.py`, `measurements.py` | Shared spike/bootstrap metrics; raw measurement parsing |
| `forecasting/oad_features.py` | 16 OAD features (leak-safe R-5 day lag; cloud fractions) |
| `forecasting/raw_model_factory.py`, `classification_adapter.py` | Model builders; categories and the spike classifier |
| `backend/api.py` | FastAPI: forecasts, retrospective, visualizations, `/api/metrics/summary` |
| `frontend/` | React + Vite dashboard (trimmed Plotly bundle in `src/utils/Plot.js`) |
| `dataset-creation.py` | Full dataset build (downloads; see hard rule 1) |
| `scripts/eval/`, `scripts/tune/`, `scripts/data/` | Evaluation, tuning, offline data tools |
| `oad/` | Ocean Anomaly Detection subproject (MODIS autoencoder; see `oad/README.md`) |
| `papers/` | `datect/`, `oad_mdpi/`, `oad_cvpr/`, `poster/` |

**Models:** per-site weighted blend of XGBoost + Random Forest (`ensemble`); naive persistence
is an external baseline; Ridge/logistic (`linear`) is a competitor. Training uses real plus
gap-filled DA rows; evaluation uses real raw measurements only.

**OAD integration:** the AE checkpoint `ae_3d_l32_c4_t4_s42_mae070` scores 5 PNW regions;
each site maps to one region. Scores are lagged so the centered 8-day composite ends before
the anchor (score at `test_date - 12`), and the in-region valid-pixel fraction is a parallel
feature because the score correlates with cloud cover (r ~ 0.4-0.5).

## Results

All current numbers, their protocol and history are in `docs/CORRECTED_NUMBERS.md`.
<!-- HEADLINE: filled after the post-cleanup re-run -->

## Docs

`README.md` (setup) · `docs/CORRECTED_NUMBERS.md` (all numbers) ·
`docs/QUICKSTART_FOR_REVIEWERS.md` (reproducing claims, script index) ·
`docs/PIPELINE_DEEP_DIVE.md` · `docs/DATA_PIPELINE_DETAILED.md` ·
`docs/OAD_INTEGRATION_RESULTS.md` · `docs/HYAK_SETUP.md` · `docs/DEPLOYMENT_GUIDE.md` ·
`docs/VISUALIZATIONS_GUIDE.md` · `oad/README.md`, `oad/RESULTS.md`.

Before pasting a file path from memory or an old note into an instruction, check it exists.

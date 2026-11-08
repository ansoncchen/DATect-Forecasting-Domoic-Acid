# Engine cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the duplicate and dead prediction paths from `forecasting/` and `backend/` without changing any prediction, prove it with the Hyak golden gate, and then rebuild the precomputed store once.

**Architecture:**
- **Deletions.** The engine keeps one per-point forecast function, used by the retrospective, the Spark tuner and the dashboard's single forecast. It keeps one ensemble blend and one raw-DA reader. Removed: the 4-category classifier, the logistic path, the quantile and bootstrap intervals (the dashboard shows the conformal band), per-anchor tuning, and endpoints the new frontend never calls.
- **Gate.** Every prediction column must come out byte-identical. The only allowed output change is the always-empty `predicted_category_logistic` column, which `compare_golden.py` must be told about explicitly.

**Tech Stack:** Python 3.10 (pinned on Hyak and CI), pandas, XGBoost, scikit-learn, FastAPI, pytest, PySpark (store builder only).

**Spec:** the duplicate-systems audit of 2026-11-03 (batch 1, engine half), and its code map of 2026-11-07, both summarised under Context. Also `CLAUDE.md` hard rule 4: behavior-neutral refactors must reproduce the golden baseline exactly.

## Context (what the audit and the code map found)

- **Intervals** (`forecasting/raw_forecast_engine.py`):
  - (a) `_compute_confidence_intervals` (≈415-486): quantile XGBoost plus RF tree spread.
  - (b) `generate_bootstrap_confidence_intervals` (≈488-581): global NumPy RNG.
  - (c) a p10/p50/p90 block (≈1059-1077) behind `skip_quantiles`, which every caller sets to True.

  None feeds a prediction or a retrospective column. The dashboard shows the conformal band instead.
- **The 4-category classifier and the logistic path** (engine ≈343-395, ≈605, ≈1042-1057; `classification_adapter.py`): no current caller uses them. The frontend sends only `task: "regression"` and `model: "ensemble"`, and golden scripts use only `task="regression"`.
- **Per-anchor tuning is off and unreachable.** Every site's grid has 2-4 entries. `_run_single_retrospective` (≈816-907) therefore amounts to site params plus the core call. `generate_single_forecast` applies site params and calls the core itself.
- **The blend `w_xgb·xgb + w_rf·rf (+ w_naive·naive)` is written in four engine places,** with defaults (0.5, 0.5, 0) in the engine and (0.56, 0.44, 0) in `per_site_models.py`. All 10 sites have JSON weights, so neither default is reached.
- **The frontend never calls** `/api/forecast`, `/api/retrospective` or `/api/models`. It still calls `/api/sites`, `/api/config`, `/api/forecast/enhanced`, `/api/forecast/all`, `/api/sites/{site}/history`, `/api/metrics/summary`, `/api/track-record` and `/api/visualizations/*`. `/api/historical` stays: it is small and correct.
- **The store's `retrospective` table** has 8 slices. With `/api/retrospective` gone, only `regression/xgboost` and `regression/rf` are read, by spectral analysis.
- **The site map** reads `get_latest_da_from_raw_files` in `backend/api.py` (≈586-644). It takes each CSV's last row, which depends on file order; the API's canonical reader is `_raw_measurements()`.
- **Broken or obsolete files:**
  - `datect_spark/shared_vs_per_site.py` calls a `retune._apply` that doesn't exist;
  - `datect_spark/retune.py` still requires `DATECT_CLIP_Q_OVERRIDE=none`;
  - `tune_all.py` is still named in `forecasting/tuned_config.py`, `forecasting/per_site_models.py` and the tuned JSON's `description`.

## Global Constraints

- **No prediction may change.** Every column of every golden prediction parquet must be byte-identical (`compare_golden.py`, `check_exact=True`). The only allowed difference is dropping the always-empty `predicted_category_logistic` column.
- **Keep XGBoost's thread settings as they are:** `n_jobs=1` in the retrospective and the default in single forecasts. Equality across thread counts was only shown empirically.
- **Keep `forecasting/oad_features.py`, `forecasting/conformal.py` and `forecasting/metrics.py` as they are** (new uses only).
- **Keep the store's table schemas:** the `forecasts` table keeps its `task` column, always `"regression"`.
- **Leave alone:** `frontend/` (except removing fixture keys if a test needs it), `papers/`, and the untracked files under `papers/`.
- **Never re-download ERDDAP data.** Heavy runs happen on Hyak only, with no SSH polling loops.
- **Every task ends with:** `.venv/bin/python -m pytest tests/ -q` green, and `ruff check` adding no new findings in the files touched.

## Review Focus

1. **`model="linear"`, `"rf"` or `"naive"` requests** to `/api/forecast/enhanced` and `/all` still work once the bootstrap code is gone. A reasonable person expects a forecast with no band (`band_reason: "calibrated for the ensemble only"`), not a 500. Covered in Task 6.
2. **A `task` other than `"regression"`,** or `model_type="logistic"`, sent to the engine fails loudly with `ValueError`, not with a silent fall-through to XGBoost. Covered in Task 5.
3. **`DATECT_USE_PER_SITE_MODELS=false`** still blends 0.5/0.5 everywhere: retrospective, single forecast and spike alert. Covered in Task 4.
4. **Scripts that relied on the deleted pieces** fail clearly, not silently. That means `chronological_eval.py --task classification`, `paper_sensitivity_study.py` (which uses `DATECT_CLIP_Q_OVERRIDE` numbers, which stay) and `scripts/eval/endpoint_bodies.py`. Covered in Tasks 5 and 7.
5. **The site map with a site that has no measurements** still renders that site as no data. Covered in Task 8.

---

### Task 1: Let the golden comparison allow named, declared drops

**Files:**
- Modify: `scripts/eval/compare_golden.py` (`compare` and the CLI)
- Test: `tests/test_compare_golden.py` (new)

**Interfaces:**
- Produces:
  - `compare(base: Path, cand: Path, allow_dropped_columns: frozenset[str] = frozenset(), allow_missing: frozenset[str] = frozenset()) -> int`, returning 0 when everything is identical or allowed;
  - CLI flags `--allow-dropped-columns COL[,COL]` and `--allow-missing RELPATH[,RELPATH]`.
- Each allowance is printed as `DROPPED (allowed): <file>: <col>` or `MISSING (allowed): <file>`. A column missing from the candidate that isn't listed, a column added in the candidate, or any value change still fails.

- [ ] **Step 1: Write the failing tests.** Use temp dirs holding one parquet under `results/chronological/chronological_regression_ensemble_x.parquet`, which matches `GOLDEN_GLOBS`.
  - `test_identical_runs_pass`
  - `test_dropped_column_fails_unless_allowed`: the candidate lacks `predicted_category_logistic`. It returns nonzero with no flag, and 0 with `allow_dropped_columns={"predicted_category_logistic"}`.
  - `test_value_change_fails_even_with_allowances`
  - `test_added_column_fails`
  - `test_missing_file_fails_unless_allowed`
- [ ] **Step 2: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/test_compare_golden.py -q`
Expected: FAIL (`compare` takes no allowances).
- [ ] **Step 3: Implement.** Before comparing exactly, drop the allowed columns from the baseline frame only when the candidate lacks them.
- [ ] **Step 4: Run tests**

Run the same command. Expected: PASS.
- [ ] **Step 5: Commit** with the message `golden: allow declared column drops and missing files, printed explicitly`.

### Task 2: Delete dead code and the endpoints nothing calls

**Files:**
- Modify:
  - `forecasting/raw_forecast_engine.py`: drop the p10/p50/p90 block, the `skip_quantiles` parameter and its three callers' argument, and `self._data_cache`;
  - `backend/api.py`: delete `POST /api/forecast` with `ForecastResponse`, `POST /api/retrospective` with `RetrospectiveRequest` and `_compute_summary`, and `GET /api/models` with `ModelInfo`. Also delete `get_model_factory` if nothing else uses it.
- Delete:
  - `datect_spark/shared_vs_per_site.py` (it calls a function that doesn't exist; `results/tuning_provenance/retune_spark/shared_vs_per_site_holdout.json` stays as the record);
  - `forecasting/ensemble_model_factory.py` and the `ModelFactory` export in `forecasting/__init__.py`, if `/api/models` was their only user (grep first).
- Text edits:
  - remove the `tune_all.py` mentions in `forecasting/tuned_config.py`, `forecasting/per_site_models.py` and the tuned JSON's `description`;
  - remove `datect_spark/retune.py`'s `DATECT_CLIP_Q_OVERRIDE=none` guard and setting (lines ≈90-91, ≈133), which are no-ops now that every clip is null.
- Tests:
  - update `tests/test_precomputed.py`: drop the `/api/forecast` and `/api/retrospective` endpoint tests;
  - update `tests/test_api_robustness.py`: drop the `/api/models` test, and replace `test_region_rollup_puts_clatsop_and_cannon_in_oregon` with `test_site_states_follow_the_columbia_border` (`config.SITE_STATES` stays);
  - add `tests/test_dead_endpoints.py`.

- [ ] **Step 1: Write the failing test** `test_endpoints_the_dashboard_never_calls_are_gone`: `POST /api/forecast` gives 404 or 405, `POST /api/retrospective` gives 404 or 405, and `GET /api/models` gives 404. The ones the frontend uses still answer (`/api/config` gives 200).
- [ ] **Step 2: Run it and see it fail**

Run: `.venv/bin/python -m pytest tests/test_dead_endpoints.py -q`
Expected: FAIL (they still answer 200).
- [ ] **Step 3: Delete** the code listed above. Grep for every name you delete, across `backend forecasting datect_spark scripts tests docs`, and fix or remove each reference.
- [ ] **Step 4: Run tests**

Run the full suite. Expected: PASS.
- [ ] **Step 5: Commit** with the message `cleanup: drop dead endpoints, the unused p10/p50/p90 block and stale tuner remnants`.

### Task 3: One per-point forecast function; per-anchor tuning removed

**Files:**
- Modify `forecasting/raw_forecast_engine.py`:
  - rename `_run_single_retrospective` to `_point_forecast` and drop its tuning branch, `_tune_xgb_params` and the redundant `get_site_training_frame` call;
  - make `generate_single_forecast` call `_point_forecast` instead of applying site params and calling the core itself.
- Modify `config.py`: remove `PER_ANCHOR_TUNING`, `MIN_TRAINING_FOR_TUNING`, the param-grid and calibration settings (≈346-356, ≈378-386), and the `DATECT_PER_ANCHOR_TUNING` / `DATECT_MIN_TRAINING_FOR_TUNING` reads.
- Modify `config/tuned_hyperparameters.json`: remove the global `min_training_for_tuning`, `calibration_fraction`, `max_calibration_rows`, `min_tuning_samples` and `param_grid_global` (whichever exist), each site's `param_grid`, and their provenance notes.
- Modify `forecasting/per_site_models.py`: remove `get_site_param_grid`.
- Modify `datect_spark/retune.py`: call `_point_forecast`.
- Tests: update `tests/conftest.py`, `tests/test_tuned_config.py` (required keys) and `tests/test_site_parameterized.py`.

**Interfaces:**
- Produces `RawForecastEngine._point_forecast(self, raw_measurement, feature_frame, base_params, n_jobs: int | None = 1, return_internals: bool = False)`. It applies site XGB params when `config.USE_PER_SITE_MODELS`. It sets `n_jobs` on the XGB params only when not None, then calls `_run_single_raw_validation`.
- The retrospective and `retune.py` pass `n_jobs=1`. `generate_single_forecast` passes `n_jobs=None`, keeping its current threads.

- [ ] **Step 1: Write the failing tests:**
  - `test_per_anchor_tuning_is_gone`: `config` has no `PER_ANCHOR_TUNING`, `RawForecastEngine` has no `_tune_xgb_params`, and no site block in the tuned JSON has `param_grid`.
  - `test_single_forecast_goes_through_the_point_function`: use `fast_models` and monkeypatch-spy `_point_forecast`. One `generate_single_forecast(..., "Newport", "regression", "ensemble")` calls it once, with `n_jobs=None`.
- [ ] **Step 2: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/test_tuned_config.py tests/test_forecasting_pipeline.py -q -k "per_anchor or point_function"`
Expected: FAIL.
- [ ] **Step 3: Implement.** Grep `PER_ANCHOR_TUNING|_tune_xgb_params|param_grid|MIN_TRAINING_FOR_TUNING|_run_single_retrospective` and fix every hit, docs included: `CLAUDE.md` (the per-anchor bullet and the env-var table), `README.md`, `docs/HYAK_SETUP.md`.
- [ ] **Step 4: Run tests**

Run the full suite. Expected: PASS.
- [ ] **Step 5: Commit** with the message `engine: one per-point forecast function; remove per-anchor tuning (off and unreachable)`.

### Task 4: One ensemble blend

**Files:**
- Modify:
  - `forecasting/per_site_models.py`: add `blend`; make `get_site_ensemble_weights(site)` return `(0.5, 0.5, 0.0)` when per-site models are off or the site has no weights, as the only default;
  - `forecasting/raw_forecast_engine.py`: use `blend` in the retrospective (≈723-751), the single forecast (≈291-296) and the spike alert (≈1120-1122, ≈1130-1132); make `_ensemble_weights` delegate.
- Test: `tests/test_per_site_models.py` (new) or `tests/test_site_parameterized.py`

**Interfaces:**
- Produces `blend(xgb: float, rf: float | None, naive: float, weights: tuple[float, float, float]) -> float` in `forecasting/per_site_models.py`.
  - It evaluates `w_xgb * xgb + w_rf * (xgb if rf is None else rf) + w_naive * naive` left to right.
  - The spike alert calls it with `naive=0.0`, keeping its current two-term value: `x + 0.0 == x` for finite x.

- [ ] **Step 1: Write the failing tests:**
  - `test_blend_matches_the_engines_formulas_bitwise`: for 1,000 random draws (seed 0) of the three predictions and every site's JSON weights, compare `blend` with the two current inline expressions (with and without the naive term) using `==`, not approx.
  - `test_equal_weights_when_per_site_models_are_off`: with `fresh_import` and `DATECT_USE_PER_SITE_MODELS=false`, `get_site_ensemble_weights("Newport") == (0.5, 0.5, 0.0)`.
- [ ] **Step 2: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/test_per_site_models.py -q`
Expected: FAIL (no `blend`).
- [ ] **Step 3: Implement**, replacing each inline expression with a `blend` call that keeps the operand order.
- [ ] **Step 4: Run tests**

Run the full suite. Expected: PASS.
- [ ] **Step 5: Commit** with the message `engine: one ensemble blend and one default weight`.

### Task 5: Remove the 4-category classifier and the logistic path

**Files:**
- Modify `forecasting/raw_forecast_engine.py`:
  - remove the classification and `both` branch of `generate_single_forecast` (≈343-395);
  - remove `_retro_model_type`, the per-point LogisticRegression (≈1042-1057) and the `predicted_category_logistic` output;
  - `task != "regression"` and `model_type == "logistic"` raise `ValueError` in both `generate_single_forecast` and `run_retrospective_evaluation`.
- Modify `forecasting/classification_adapter.py`: remove `train_dedicated_classifier`, `predict_with_probabilities` and `classify_prediction`. Keep `threshold_classify`, `threshold_classify_series`, `train_spike_binary_classifier` and `predict_spike_probability`.
- Modify `forecasting/raw_model_factory.py`: remove `build_logistic_classifier`, and `build_xgb_classifier` if only `tests/test_model_seed.py` uses it.
- Modify `config.py`: remove `XGB_CLASSIFICATION_PARAMS` if unused.
- Modify `backend/api.py`: `ForecastRequest.task: Literal["regression"] = "regression"`.
- Modify `scripts/eval/chronological_eval.py`: `--task` accepts `regression` only.
- Tests: update `tests/test_forecasting_pipeline.py` (≈86-103), `tests/test_site_parameterized.py` (classification cases), `tests/test_model_seed.py` and `tests/test_forecast_payload.py`.

- [ ] **Step 1: Write the failing tests:**
  - `test_classification_task_is_rejected`: with `fast_models`, `generate_single_forecast(..., task="classification")` raises `ValueError`, and so does `run_retrospective_evaluation(task="regression", model_type="logistic")`.
  - `test_retrospective_has_no_logistic_column`: `run_retrospective_evaluation(task="regression", model_type="ensemble", max_test_samples=3)` returns columns that exclude `predicted_category_logistic` and include `predicted_da_linear`.
  - `test_enhanced_rejects_classification`: `POST /api/forecast/enhanced` with `task: "classification"` gives 422.
- [ ] **Step 2: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/test_forecasting_pipeline.py tests/test_forecast_payload.py -q -k "classification or logistic"`
Expected: FAIL.
- [ ] **Step 3: Implement.** Then grep `classification|logistic|predicted_category_logistic|train_dedicated_classifier` across `backend forecasting datect_spark scripts tests docs`. Keep everything about the spike classifier and the threshold categories.
- [ ] **Step 4: Run tests**

Run the full suite. Expected: PASS.
- [ ] **Step 5: Commit** with the message `engine: remove the 4-category classifier and the logistic path (no caller; drops an always-empty column)`.

### Task 6: Remove the quantile and bootstrap intervals

**Files:**
- Modify `forecasting/raw_forecast_engine.py`: remove `_compute_confidence_intervals`, `generate_bootstrap_confidence_intervals`, their call (≈334-341), the global `random.seed`/`np.random.seed` (≈131-132) if nothing else needs them (grep), and the `"threshold"` model alias.
- Modify `config.py`: remove `ENABLE_BOOTSTRAP_INTERVALS`, `N_BOOTSTRAP_ITERATIONS`, `BOOTSTRAP_SUBSAMPLE_FRACTION`, `CONFIDENCE_PERCENTILES` and `ENABLE_QUANTILE_INTERVALS`.
- Modify `datect_spark/forecast_serving.py`: remove `OUT_SCHEMA`'s q05/q50/q95, the q fields in `_summary`, and `not_reproducible` and its use.
- Tests: update `tests/conftest.py` (bootstrap patch), `tests/test_forecasting_pipeline.py` (≈80-83), `tests/test_precomputed.py` (`_result` fixture) and `tests/test_forecast_serving.py` (drop `test_bootstrap_interval_is_not_stored`).

- [ ] **Step 1: Write the failing tests:**
  - `test_single_forecast_carries_no_interval`: with `fast_models`, `"bootstrap_quantiles" not in generate_single_forecast(..., "Newport", "regression", "ensemble")`.
  - `test_every_model_still_forecasts_without_a_band`: through the API with a stub engine, `model` ∈ {linear, rf, naive, xgboost} each returns `success: true`, `band: null` and `band_reason: "calibrated for the ensemble only"`.
  - `test_interval_knobs_are_gone`: `config` has none of the five names above.
- [ ] **Step 2: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/test_forecasting_pipeline.py tests/test_forecast_payload.py -q -k "interval or without_a_band or knobs"`
Expected: FAIL on the first and third tests.
- [ ] **Step 3: Implement**, then grep `bootstrap_quantiles|q05|q95|BOOTSTRAP|QUANTILE_INTERVALS|CONFIDENCE_PERCENTILES|threshold"` and fix each hit.
- [ ] **Step 4: Run tests**

Run the full suite. Expected: PASS.
- [ ] **Step 5: Commit** with the message `engine: remove the quantile and bootstrap intervals (the dashboard shows the conformal band)`.

### Task 7: The store keeps only the retrospective slices spectral reads

**Files:**
- Modify:
  - `datect_spark/precompute_store.py`: `SLICES = (("regression", "xgboost"), ("regression", "rf"))`;
  - `backend/precomputed.py`: remove `table_info`, which only `/api/retrospective` used, and update the module docstring;
  - `scripts/eval/endpoint_bodies.py`: drop the retrospective endpoint and its slices.
- Docs: `docs/DEPLOYMENT_GUIDE.md`, `docs/DATA_ENGINEERING.md`, `datect_spark/RESULTS.md` (note the change; keep the old measurements as history), `README.md`, `CLAUDE.md`.
- Test: `tests/test_precomputed.py`

- [ ] **Step 1: Write the failing test** `test_store_builds_only_the_slices_spectral_reads`. It asserts `precompute_store.SLICES == (("regression", "xgboost"), ("regression", "rf"))`. If importing `datect_spark` needs Spark, read the assignment from the module's source with `ast`.
- [ ] **Step 2: Run it and see it fail**

Run: `.venv/bin/python -m pytest tests/test_precomputed.py -q -k slices`
Expected: FAIL.
- [ ] **Step 3: Implement**, and check that `check_store` still requires all four tables and the spectral endpoints still read `regression/xgboost` and `regression/rf`.
- [ ] **Step 4: Run tests**

Run the full suite. Expected: PASS.
- [ ] **Step 5: Commit** with the message `store: build only the retrospective slices spectral analysis reads`.

### Task 8: One raw-DA reader for the API

**Files:**
- Modify `backend/api.py`: `get_site_map` takes each site's latest row by date from `_raw_measurements()`; delete `get_latest_da_from_raw_files`.
- Test: `tests/test_api_robustness.py`

- [ ] **Step 1: Write the failing tests:**
  - `test_map_uses_the_canonical_measurements`: for each site, the map's hover text contains the loader's latest value (`f"{value:.2f}"`) and its date (`YYYY-MM-DD`), taken from `load_raw_da_measurements()` grouped by site at the max date.
  - `test_map_marks_a_site_without_measurements`: monkeypatch `_raw_measurements` to drop one site. That site's hover says `no data` and its marker is `#d0d5db`.
- [ ] **Step 2: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/test_api_robustness.py -q -k map`
Expected: FAIL (`_raw_measurements` isn't used by the map yet).
- [ ] **Step 3: Implement**, and delete the old reader.
- [ ] **Step 4: Run tests**

Run the full suite. Expected: PASS.
- [ ] **Step 5: Commit** with the message `api: the site map reads the canonical measurements (one raw-DA reader)`.

### Task 9: Docs sweep and full check

**Files:**
- Modify:
  - `CLAUDE.md`: the single-forecast bullet (no engine intervals; `linear` is now deterministic), the per-anchor bullet and env-var rows, the architecture table, and the precomputed-store bullet;
  - `docs/PIPELINE_DEEP_DIVE.md` §9;
  - `docs/VISUALIZATIONS_GUIDE.md`;
  - `README.md`.

- [ ] **Step 1: Find stale names.**

Run: `grep -rn "bootstrap_quantiles\|_run_single_retrospective\|_tune_xgb_params\|PER_ANCHOR_TUNING\|train_dedicated_classifier\|predicted_category_logistic\|/api/retrospective\|/api/models\|get_latest_da_from_raw_files\|ENABLE_QUANTILE_INTERVALS" --include='*.md' --include='*.py' . | grep -v "docs/superpowers\|CORRECTED_NUMBERS\|FORECAST_EXPERIMENTS\|\.venv\|node_modules"`
Expected: no hits outside history sections.
- [ ] **Step 2: Fix the docs.**
- [ ] **Step 3: Run the full suite, lint and the frontend checks.**

Run: `.venv/bin/python -m pytest tests/ -q && (cd frontend && npm run lint && npm run build && npm test -- --run)`
Expected: all pass.
- [ ] **Step 4: Commit** with the message `docs: engine cleanup`, then push the branch.

### Task 10: The golden gate on Hyak

This task runs on Hyak and needs the SSH ControlMaster. If it is down, ask the user to run `ssh klone-login 'whoami'`.

- [ ] **Step 1: Run two golden jobs**, one at `main` (`c684c265`) and one at the branch head. Use `scripts/eval/golden_baseline.sh <ref> <label>` with labels `engine-cleanup-base` and `engine-cleanup-head`.
  - If `git worktree` fails with "Disk quota exceeded" (scrubbed is shared and near full), stage each with `git archive <ref> | tar -x` into `$HOME/<label>` and `sbatch --chdir` it there, as was done for the band run.
  - Wait with one server-side loop, never polling.
- [ ] **Step 2: Compare.**

Run on Hyak: `python scripts/eval/compare_golden.py <base> <head> --allow-dropped-columns predicted_category_logistic`
Expected: exit 0. Every prediction file is identical except the allowed dropped column, and the metric tables are identical or within 1e-12.
- [ ] **Step 3: Record the result** (job IDs, the compare output summary) in the PR description. Open the PR; the user merges it.

**After merge (not part of this plan):** rebuild the precomputed store once at the merged commit (`datect_spark/store.sbatch`, all steps). That unblocks deploys. Then the user re-enables Cloud billing and the current version is deployed.

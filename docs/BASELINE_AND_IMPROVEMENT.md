# Baseline Metrics and “Did We Improve?”

## Short answer

**Yes.** Regression R² was improved by applying the **best hyperparameters** from `cache/hyperparams/regression_xgboost_best.json` (from running `forecasting.hyperparam_search`). Baseline config gave R² **~0.47**; with best params R² is **~0.50** (same protocol: 500 anchors, min_test_date 2008-01-01). Linear and logistic remain baselines to show XGBoost superiority. CatBoost/LightGBM cache artifacts were removed; only the four canonical task/model pairs are used.

## Canonical baseline (main)

These numbers are the **single source of truth**. Any change must be compared against them using the **exact same evaluation protocol** below.

| Task / Model        | Metric   | Baseline value |
|---------------------|----------|----------------|
| Regression (linear) | R²       | **0.1056**     |
| Regression (XGBoost)| R²       | **0.4729**     |
| Classification (logistic) | Accuracy | **0.674**  |
| Classification (XGBoost) | Accuracy | **0.8184**  |

## Evaluation protocol (reproduce baseline)

- **Source:** Retrospective evaluation (same logic as `precompute_cache.py` on main).
- **Task/model pairs:**  
  `(regression, xgboost)`, `(regression, linear)`, `(classification, xgboost)`, `(classification, logistic)`.
- **Settings:**  
  `n_anchors = config.N_RANDOM_ANCHORS` → 500  
  `min_test_date = "2008-01-01"`  
  Data: `config.FINAL_OUTPUT_PATH` → `./data/processed/final_output.parquet`
- **Metrics:**  
  - Regression: `r2_score(actual_da, predicted_da)`, `mean_absolute_error(actual_da, predicted_da)`.  
  - Classification: `accuracy_score(actual_category, predicted_category)` on the full retrospective result list.
- **Config (main):**  
  `FORECAST_HORIZON_WEEKS = 1`, `FORECAST_HORIZON_DAYS = 7`, `RANDOM_SEED = 42`, `MIN_TRAINING_SAMPLES = 3`,  
  `USE_LAG_FEATURES = False`, `LAG_FEATURES = []`, `USE_ROLLING_FEATURES = False`,  
  `USE_ENHANCED_TEMPORAL_FEATURES = True`, `USE_REGRESSION_SAMPLE_WEIGHTS = True`,  
  XGBoost regression: `n_estimators=400`, `max_depth=6`, `learning_rate=0.05`, etc.  
  Linear: `LinearRegression(n_jobs=-1)`; **no sample weights** in the single-anchor fit path.  
  Logistic: `LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, random_state=42, n_jobs=-1)` with classification sample weights when supported.

## Cleanup done (artifact removal)

- **Config:** Reverted to main-equivalent: `USE_LAG_FEATURES=False`, `USE_ROLLING_FEATURES=False`, `USE_SITE_ENCODING=False`, `USE_LOG_TARGET_TRANSFORM=False`; XGB params set to main defaults; removed LightGBM/CatBoost config blocks.
- **Precompute:** Only the four task/model pairs above; no LightGBM, CatBoost, or stacking.
- **model_factory.py:** Only XGBoost and linear/logistic; removed LightGBM, CatBoost, and stacking.
- **requirements.txt:** Removed `lightgbm`, `catboost`, `torch`; kept `xgboost` and scikit-learn.
- **.gitignore:** Added `catboost_info/` (ML artifact junk).

`forecasting/neural_model.py` is **not** used anywhere in the main pipeline; it remains in the repo as experimental only.

## How to improve from here

1. **Lock this protocol** — Use the same retrospective script, same 500 anchors, same horizon and dates for every comparison.
2. **One change at a time** — Toggle a single feature (e.g. `USE_LAG_FEATURES=True` with `LAG_FEATURES=[1]` only for DA) or one hyperparameter, re-run retrospective, compare R²/accuracy to the table above.
3. **Hyperparameter search** — Run `python3 -m forecasting.hyperparam_search --task regression --trials 100 --anchors 500`. Best params are saved to `cache/hyperparams/`; ModelFactory loads them when present (config is fallback).
4. **Optional lag-1** — Enabling `USE_LAG_FEATURES=True` and `LAG_FEATURES=[1]` adds “last week’s DA” as a feature; test with `scripts/measure_regression_r2.py` (N_ANCHORS=100 for quick, 500 for full).
5. **Raw vs engineered data** — Testing on raw or minimal features may help; use the same protocol so results are comparable.
6. **Quick checks** — `scripts/measure_regression_r2.py` with `N_ANCHORS=100` for fast iteration or `N_ANCHORS=500` for full comparison.

## Experimental / unused code

- **`forecasting/neural_model.py`** — Not used in the main pipeline; kept for experimentation only. Remove or move to an `experimental/` folder if simplifying the codebase.

## Pipeline modernization (future)

- **Current:** Pandas + Parquet + optional DuckDB for reads. NetCDF for satellite ingestion; Python-only.
- **Options to consider:** Polars for faster tabular ops, DuckDB for SQL-over-Parquet in more places, and keeping the same high-level flow (dataset-creation → final_output.parquet → forecast_engine). Full rewrite in another language is a larger effort; incremental modernization (e.g. Polars in dataset-creation or data_processor) can be done while preserving temporal integrity and evaluation protocol.

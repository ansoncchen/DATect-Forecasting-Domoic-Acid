# DATect Fix Plan - Ralph Task List

## Current Metrics to Beat
- XGBoost Regression R²: 0.473 → Target: **> 0.55**
- XGBoost Classification Accuracy: 81.8% → Target: **> 85%**

---

## Phase 1: Model Performance Improvement (HIGH PRIORITY)

### 1.1 Hyperparameter Optimization
- [ ] Implement Bayesian optimization for XGBoost hyperparameters (use Optuna or scikit-optimize)
- [ ] Set up proper temporal cross-validation (TimeSeriesSplit or custom walk-forward)
- [ ] Test different learning rate schedules (0.01, 0.03, 0.05, 0.1)
- [ ] Experiment with tree depth (5-10 range) and n_estimators (300-800)
- [ ] Document best hyperparameters found with their validation scores

### 1.2 Feature Engineering Experiments
- [ ] Re-evaluate USE_LAG_FEATURES with proper temporal CV (was disabled - test if it helps now)
- [ ] Re-evaluate USE_ROLLING_FEATURES with proper temporal CV
- [ ] Test interaction features between satellite data and temporal features
- [ ] Experiment with different DA_CATEGORY_BINS thresholds
- [ ] Test log transformation of target variable (domoic acid concentrations)
- [ ] Compare raw data vs engineered features performance

### 1.3 Ensemble & Model Refinement
**Historical testing (Aug 2025)**: XGBoost (R²=0.839), Stacking Ensemble (R²=0.845), LightGBM (R²=0.832), Random Forest (R²=0.781) were top performers. But dataset/implementation may have changed since then.

- [ ] Implement stacking ensemble (XGBoost + LightGBM + Random Forest) - historically best at R²=0.845
- [ ] Test weighted averaging of XGBoost + Random Forest predictions
- [ ] Experiment with blending approaches
- [ ] Consider adding CatBoost to ensemble (R²=0.824 historically)
- [ ] Re-test LightGBM with current dataset/evaluation (historically performed well)
- [ ] If time permits and you have a hypothesis, re-test one "failed" model to verify past results still hold

### 1.4 Model Diagnostics
- [ ] Analyze feature importances from current XGBoost model
- [ ] Identify which features contribute most/least to predictions
- [ ] Check for any remaining data leakage issues
- [ ] Analyze prediction errors by site, season, and DA level

---

## Phase 2: Data Pipeline Modernization (MEDIUM PRIORITY)

### 2.1 Performance Improvements
- [ ] Profile current dataset-creation.py to identify bottlenecks
- [ ] Evaluate Polars as Pandas replacement for data processing
- [ ] Evaluate DuckDB for analytical queries
- [ ] Implement parallel processing where applicable
- [ ] Add progress bars and timing logs

### 2.2 Code Quality
- [ ] Add data validation checks at each pipeline stage
- [ ] Implement proper error handling with informative messages
- [ ] Add logging throughout the pipeline
- [ ] Create data quality reports (missing values, outliers, distributions)

### 2.3 Architecture
- [ ] Refactor into modular ETL stages (Extract, Transform, Load)
- [ ] Add configuration file for pipeline parameters
- [ ] Implement caching for intermediate results
- [ ] Add dry-run mode for testing pipeline changes

---

## Phase 3: Codebase Cleanup (LOWER PRIORITY)

### 3.1 Code Organization
- [ ] Audit codebase for dead/unused code
- [ ] Consolidate duplicate functions across files
- [ ] Improve file/folder organization
- [ ] Add __init__.py files where missing

### 3.2 Documentation
- [ ] Add docstrings to all public functions
- [ ] Add type hints throughout codebase
- [ ] Update CLAUDE.md with any new commands or workflows
- [ ] Create architecture diagram

### 3.3 Technical Debt
- [ ] Simplify overly complex functions (>50 lines)
- [ ] Remove commented-out code blocks
- [ ] Standardize naming conventions
- [ ] Add unit tests for critical functions

---

## Completed Tasks
- [x] Project enabled for Ralph
- [x] Initial PROMPT.md configured with project goals
- [x] fix_plan.md created with prioritized tasks

---

## Experiment Log

### Template for Recording Results
```
Date: YYYY-MM-DD
Experiment: [brief description]
Changes: [what was modified]
Results:
  - Regression R²: X.XXX (baseline: 0.473)
  - Classification Accuracy: XX.X% (baseline: 81.8%)
Conclusion: [keep/revert and why]
```

### Experiments Conducted
(Ralph will update this section with results)

---

## Notes & Learnings

- Many features were disabled because they DECREASED performance - be cautious when re-enabling
- Data is noisy and spikey - real-world HAB data with gaps that were filled
- ~10,000 data points total after filling gaps
- Always run `python3 verify_temporal_integrity.py` after changes
- Measure with `python3 precompute_cache.py` (n_anchors=500, min_test_date=2008-01-01)

---

## Blocked Items
(Items waiting on external dependencies or decisions)

- [ ] Hyak cluster testing - not available yet for large-scale experiments

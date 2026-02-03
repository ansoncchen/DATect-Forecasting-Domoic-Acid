# DATect Forecasting System - Ralph Development Instructions

## Context
You are Ralph, an autonomous AI development agent working on **DATect**, a machine learning system for forecasting harmful algal bloom toxin concentrations (domoic acid) along the Pacific Coast.

**Project Type:** Python (ML/Data Science)

The primary mission is to demonstrate that ML approaches significantly outperform traditional statistical methods for this critical public health forecasting task.

---

## Priority 1: Improve Model Performance (CRITICAL)

### Current Baseline Metrics (YOU MUST BEAT THESE)
| Model | Task | Metrics | Runtime | Target |
|-------|------|---------|---------|--------|
| Linear Regression | Regression | R² = 0.106 | - | Baseline only |
| **XGBoost** | **Regression** | **R² = 0.467, MAE = 5.48, F1 = 0.686** | **3.1 min** | **R² > 0.55** |
| Logistic Regression | Classification | Accuracy = 67.4% | - | Baseline only |
| **XGBoost** | **Classification** | **Accuracy = 81.8%** | **4.7 min** | **> 85%** |

### Baseline Evaluation Settings (Can Be Changed)
The baseline metrics above were calculated with these settings, but they can be modified if it improves results:

```python
# In config.py - feel free to experiment with these
N_RANDOM_ANCHORS = 500          # Number of test points for evaluation
MIN_TEST_DATE = "2008-01-01"    # Start date for test period
FORECAST_HORIZON_WEEKS = 1      # 7 days ahead
FORECAST_HORIZON_DAYS = 7
RANDOM_SEED = 42
MIN_TRAINING_SAMPLES = 3
```

Run `python3 precompute_cache.py` to evaluate model performance.
Data source: `./data/processed/final_output.parquet`

**IMPORTANT - Testing is SLOW**: Each evaluation run takes **~8 minutes total** (3.1 min regression + 4.7 min classification). Plan accordingly:
- Batch multiple changes together before running evaluation
- Don't run precompute_cache.py after every small tweak
- Make meaningful changes, then test once
- Use smaller `N_RANDOM_ANCHORS` (e.g., 50-100) for quick sanity checks during development, then full 500 for final validation

**Note**: If you change evaluation settings, document the new settings when reporting results so comparisons are fair.

### Current Model Configuration (in config.py)
```python
# Feature flags - MANY ARE DISABLED BECAUSE THEY HURT PERFORMANCE
USE_LAG_FEATURES = False
USE_ROLLING_FEATURES = False
USE_ENHANCED_TEMPORAL_FEATURES = True
USE_REGRESSION_SAMPLE_WEIGHTS = True

# Classification bins: Low(0), Moderate(1), High(2), Extreme(3)
DA_CATEGORY_BINS = [-inf, 5, 20, 40, inf]

# XGBoost Regression
XGB_REGRESSION_PARAMS = {
    "n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.85, "colsample_bytree": 0.85, "colsample_bylevel": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "gamma": 0.1, "min_child_weight": 3
}

# XGBoost Classification
XGB_CLASSIFICATION_PARAMS = {
    "n_estimators": 500, "max_depth": 7, "learning_rate": 0.03,
    "subsample": 0.9, "colsample_bytree": 0.9, "colsample_bylevel": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 2.0, "gamma": 0.2, "min_child_weight": 5
}
```

### Improvement Strategies to Explore
1. **Hyperparameter tuning** - Bayesian optimization, grid search with temporal CV (HIGHEST PRIORITY)
2. **Feature engineering** - Carefully test lag features, rolling windows (but many were disabled for a reason!)
3. **Target transformation** - Log transform DA values, different binning strategies
4. **Test on raw data** - Compare performance on raw vs. engineered features
5. **Proper temporal CV** - Ensure time-series cross-validation to avoid leakage
6. **Ensemble methods** - Combine XGBoost + Random Forest (the two best performers)

### Historical Model Testing (August 2025 - May Be Outdated)
Previous testing from **August 2025** found these results. **IMPORTANT**: Dataset, features, and evaluation may have changed since then - these results may no longer be accurate. Use as guidance, not gospel.

**PREVIOUSLY TOP PERFORMERS**
| Model | R² (Aug 2025) | Notes |
|-------|---------------|-------|
| Stacking Ensemble | 0.845 | Highest accuracy but complex |
| XGBoost | 0.839 | Best balance of accuracy/speed |
| FLAML ExtraTrees | 0.833 | AutoML optimized |
| LightGBM | 0.832 | Fast training |
| CatBoost | 0.824 | Handles categorical data |
| Random Forest | 0.781 | Good baseline, interpretable |

**PREVIOUSLY POOR PERFORMERS (But Worth Re-evaluating If You Have a Hypothesis)**
| Model | R² (Aug 2025) | Why It Failed Then |
|-------|---------------|-------------------|
| KNN | 0.523 | Sensitive to noise |
| SVM | 0.445 | Doesn't scale well |
| LSTM | 0.267 | Poor for tabular data |
| MLP | 0.234 | Overfitting issues |
| CNN | 0.198 | Inappropriate for this data |
| Transformer | 0.156 | Requires more data |
| ARIMA | -0.234 | Linear assumptions violated |
| SARIMAX | -0.156 | Can't handle multivariate |
| Prophet | -0.089 | Failed on this data |

**Key Insight**: Gradient boosting methods historically worked best. Deep learning and time series models struggled - but dataset/implementation may have changed.

**Recommendation**:
- Prioritize hyperparameter tuning and ensembles of gradient boosting models
- If re-testing other models, have a specific hypothesis for why results might differ now
- Don't blindly assume past failures will repeat - but don't waste cycles without good reason either

### CRITICAL CONSTRAINTS
- **Limited data**: ~10,000 samples with gaps, noisy real-world measurements
- **Spikey/irregular**: DA concentrations are inherently volatile and unpredictable
- **Features often HURT**: Many features are disabled because they decreased performance - be careful!
- **TEMPORAL INTEGRITY**: NEVER use future data - always run `python3 verify_temporal_integrity.py`
- **Future compute**: Hyak cluster (44K CPUs, 954 GPUs) available later but NOT for testing now

---

## Key Principles for Ralph

1. **ONE task per loop** - Focus on the most important thing
2. **Search before assuming** - Check codebase before claiming something doesn't exist
3. **Test after changes** - Always verify temporal integrity
4. **Measure performance** - Run `precompute_cache.py` after model changes
5. **Document learnings** - Update fix_plan.md with results

## Testing Guidelines
- LIMIT testing to ~20% of your total effort per loop
- PRIORITIZE: Implementation > Documentation > Tests
- Only write tests for NEW functionality you implement

## Build & Run Commands
Quick reference:
```bash
python3 run_datect.py              # Run full system
python3 verify_temporal_integrity.py   # Verify no data leakage
python3 precompute_cache.py        # Evaluate model performance (~8 min)
python3 dataset-creation.py        # Regenerate dataset (30-60 min)
```

---

## Status Reporting (CRITICAL)

At the end of EVERY response, include this status block:

```
---RALPH_STATUS---
STATUS: IN_PROGRESS | COMPLETE | BLOCKED
TASKS_COMPLETED_THIS_LOOP: <number>
FILES_MODIFIED: <number>
TESTS_STATUS: PASSING | FAILING | NOT_RUN
WORK_TYPE: IMPLEMENTATION | TESTING | DOCUMENTATION | REFACTORING
EXIT_SIGNAL: false | true
RECOMMENDATION: <one line summary of what to do next>
---END_RALPH_STATUS---
```

## Current Task
Follow fix_plan.md and choose the highest priority uncompleted item.

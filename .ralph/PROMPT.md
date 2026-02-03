# DATect Forecasting System - Ralph Development Instructions

## Context
You are Ralph, an autonomous AI development agent working on **DATect**, a machine learning system for forecasting harmful algal bloom toxin concentrations (domoic acid) along the Pacific Coast.

**Project Type:** Python (ML/Data Science)

The primary mission is to demonstrate that ML approaches significantly outperform traditional statistical methods for this critical public health forecasting task.

---

## Priority 1: Improve Model Performance (CRITICAL)

### Current Baseline Metrics (YOU MUST BEAT THESE)
| Model | Task | Metrics | Runtime | Target | Priority |
|-------|------|---------|---------|--------|----------|
| Linear Regression | Regression | R² = 0.106 | - | Baseline only | - |
| **XGBoost** | **Regression** | **R² = 0.467, MAE = 5.48, F1 = 0.686** | **3.1 min** | **R² ≥ 0.60 (MINIMUM GOAL: 0.6-0.7)** | **PRIMARY FOCUS** |
| Logistic Regression | Classification | Accuracy = 67.4% | - | Baseline only | - |
| **XGBoost** | **Classification** | **Accuracy = 81.8%** | **4.7 min** | **> 85%** | Secondary |

**🎯 CRITICAL GOAL**: You MUST achieve **R² = 0.6-0.7 minimum** for regression. This is non-negotiable. The current R² of 0.467 is insufficient.

**FOCUS**: Prioritize R² regression improvements. Classification is secondary.

### Baseline Evaluation Settings (Can Be Changed)
The baseline metrics above were calculated with these settings, but they can be modified if it improves results:

```python
# In config.py - feel free to experiment with these
N_RANDOM_ANCHORS = 500          # Number of test points for evaluation
FORECAST_HORIZON_WEEKS = 1      # 1 week ahead
FORECAST_HORIZON_DAYS = 7       # Derived from FORECAST_HORIZON_WEEKS
RANDOM_SEED = 42
MIN_TRAINING_SAMPLES = 3
```

**⚠️ MANDATORY TESTING PROCEDURE**:
```bash
python3 precompute_cache.py   # MUST use N_RANDOM_ANCHORS=500 for fair baseline comparison
```
Data source: `./data/processed/final_output.parquet`

**Testing Runtime**: Each evaluation run takes **~5 minutes total** with N_RANDOM_ANCHORS=500. This is fast enough for regular testing:
- **ALWAYS run precompute_cache.py with 500 anchors** to compare against baseline (R² = 0.467)
- Run after each meaningful model change
- Temporal integrity is automatically validated during evaluation

**Note**: Do NOT change N_RANDOM_ANCHORS from 500 - this ensures fair comparison against the baseline metrics.

### Current Model Configuration (in config.py)
```python
# Feature flags - MANY ARE DISABLED BECAUSE THEY HURT PERFORMANCE
USE_LAG_FEATURES = False
USE_ROLLING_FEATURES = False
USE_ENHANCED_TEMPORAL_FEATURES = True
USE_REGRESSION_SAMPLE_WEIGHTS = True

# Classification bins: Low(0), Moderate(1), High(2), Extreme(3)
DA_CATEGORY_BINS = [-inf, 5, 20, 40, inf]

# XGBoost Regression (verified from config.py:198-210)
XGB_REGRESSION_PARAMS = {
    "n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.85, "colsample_bytree": 0.85, "colsample_bylevel": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "gamma": 0.1, "min_child_weight": 3,
    "tree_method": "hist"
}

# XGBoost Classification (verified from config.py:213-226)
XGB_CLASSIFICATION_PARAMS = {
    "n_estimators": 500, "max_depth": 7, "learning_rate": 0.03,
    "subsample": 0.9, "colsample_bytree": 0.9, "colsample_bylevel": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 2.0, "gamma": 0.2, "min_child_weight": 5,
    "tree_method": "hist", "eval_metric": "logloss"
}
```

### Improvement Strategies - EXPERIMENT FREELY!

**You have full autonomy to try ANY approach that might improve R² regression performance.** The list below is just a starting point - don't limit yourself to these ideas:

**Suggested Directions:**
1. **Hyperparameter tuning** - Bayesian optimization, grid search with temporal CV
2. **Feature engineering** - Lag features, rolling windows, interactions, polynomial features, domain-specific transformations
3. **Target transformation** - Log transform, Box-Cox, Yeo-Johnson on DA values
4. **Different models** - Try LightGBM, CatBoost, Random Forest, stacking ensembles, neural networks if you have a hypothesis
5. **Feature selection** - Remove unhelpful features, test subsets, recursive feature elimination
6. **Data preprocessing** - Different scaling methods, outlier handling, missing value strategies
7. **Temporal features** - Better time encoding, seasonal patterns, cyclical features
8. **Ensemble methods** - Stacking, blending, weighted averaging, boosting chains
9. **Advanced techniques** - Quantile regression, multi-task learning, transfer learning from similar datasets
10. **Evaluation settings** - Experiment with forecast horizons, test periods, validation strategies

**Key Principles:**
- **Be creative** - If you have an idea, try it!
- **Document everything** - Record what you tried and why in the Experiment Log
- **Always verify temporal integrity** - No future data leakage
- **Measure impact** - Run precompute_cache.py to get R² scores
- **Keep what works** - Commit improvements, revert failures

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
- **Dataset size**: 10,950 rows in final_output.parquet (verified) with gaps filled from noisy real-world measurements
- **Spikey/irregular**: DA concentrations are inherently volatile and unpredictable
- **Features often HURT**: Many features are disabled because they decreased performance - but feel free to experiment
- **TEMPORAL INTEGRITY**: NEVER use future data - DataProcessor has built-in validation (automatically runs during evaluation)
- **Fast evaluation**: precompute_cache.py takes ~5 min locally - run after each model change

---

## Key Principles for Ralph

1. **ONE task per loop** - Focus on the most important thing
2. **Search before assuming** - Check codebase before claiming something doesn't exist
3. **Measure performance** - Run `precompute_cache.py` after model changes to get R² scores
4. **Document learnings** - Update fix_plan.md Experiment Log with results
5. **Experiment freely** - Try ANY idea that might improve R² - you're not limited to the suggested strategies
6. **Focus on regression** - R² is the primary metric, classification accuracy is secondary
7. **Temporal integrity** - Built into DataProcessor, automatically validated during evaluation

## Testing Guidelines
- LIMIT testing to ~20% of your total effort per loop
- PRIORITIZE: Implementation > Documentation > Tests
- Only write tests for NEW functionality you implement

## Build & Run Commands
Quick reference:
```bash
python3 run_datect.py              # Run full system
python3 precompute_cache.py        # Evaluate model performance (~5 min, includes temporal validation)
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

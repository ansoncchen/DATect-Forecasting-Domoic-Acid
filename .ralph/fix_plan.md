# DATect Fix Plan - Ralph Task List

## Current Metrics to Beat
- **🎯 PRIMARY GOAL**: XGBoost Regression R² = 0.467 → **MINIMUM Target: R² = 0.6-0.7** (MAE = 5.48, F1 = 0.686)
- Secondary Goal: XGBoost Classification Accuracy = 81.8% → Target: **> 85%**

**⚠️ CRITICAL**: You MUST achieve **R² ≥ 0.60** minimum. The 0.6-0.7 range is the non-negotiable goal.

**📊 MANDATORY TESTING**: Always run `python3 precompute_cache.py` with **N_RANDOM_ANCHORS=500** to compare against baseline.

**FOCUS**: Prioritize R² regression improvements. Try any approach that might work!

---

## Phase 1: Model Performance Improvement (PRIORITY)

**⚠️ IMPORTANT**: The tasks below are just IDEAS to get you started. You have COMPLETE freedom to:
- ✅ Skip any/all of these tasks if they don't seem promising
- ✅ Add your own tasks based on creative ideas
- ✅ Try completely different approaches not listed here
- ✅ Experiment with anything that might improve R² - no restrictions!
- ✅ Work on tasks in any order (or ignore the order entirely)

**Don't feel constrained by this list. It's guidance, not rules.**

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
  - Regression R²: X.XXX (baseline: 0.467, delta: +/-X.XXX)
  - Regression MAE: X.XX (baseline: 5.48)
  - Classification Accuracy: XX.X% (baseline: 81.8%) [if measured]
Conclusion: [keep/revert and why]
Next idea: [what to try next]
```

### Experiments Conducted
(Ralph will update this section with results)

---

## Notes & Learnings

- Many features were disabled because they DECREASED performance - but feel free to re-test them with different configurations
- Data is noisy and spikey - real-world HAB data with gaps that were filled
- 10,950 rows in final_output.parquet (verified actual dataset size)
- Temporal integrity is built into DataProcessor - automatically validated during precompute_cache.py
- Evaluation takes ~5 min total with N_RANDOM_ANCHORS=500 - fast enough for regular testing
- **Focus on R² regression** - that's the primary metric to improve
- Try anything that might work - you have full experimental freedom!

---

## Git Workflow Reminder

**EVERY iteration must end with a commit and push to `ralph-improvement` branch!**

```bash
git add -A
git commit -m "Ralph: [description] - R² = X.XXX"
git push origin ralph-improvement
```

This ensures all progress is tracked on GitHub for review.

---

## Remote Compute (For Reference Only)
A Hyak compute cluster is available for heavy workloads, but Ralph runs locally.
Local evaluation is fast enough (~5 min) - no remote compute needed for model experiments.

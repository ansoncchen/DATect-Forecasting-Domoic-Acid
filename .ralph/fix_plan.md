# DATect Fix Plan - Ralph Task List

## Current Metrics to Beat
- XGBoost Regression: R² = 0.467, MAE = 5.48, F1 = 0.686 (3.1 min) → Target: **R² > 0.55**
- XGBoost Classification: Accuracy = 81.8% (4.7 min) → Target: **> 85%**

---

## Phase 1: Model Performance Improvement (PRIORITY)

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
  - Regression R²: X.XXX (baseline: 0.467)
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
- Evaluation takes ~8 min total (3.1 min regression + 4.7 min classification)
- Use smaller N_RANDOM_ANCHORS (50-100) for quick sanity checks

---

## Hyak Cluster (NOW AVAILABLE)
- **Access**: `ssh ac283@klone.hyak.uw.edu`
- **Resources**: 44,184 CPU cores, 954 GPU cards
- Can be used for large-scale hyperparameter tuning, ensemble training, etc.

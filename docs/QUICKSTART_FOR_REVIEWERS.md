# Quickstart for Reviewers

> A single-page guide to verifying any claim in the DATect paper from the
> source artifacts. Each section answers one reviewer question with the
> exact commands to run and the exact files to inspect.

---

## "Where do the headline numbers come from?"

Every numeric claim in the paper traces to one of three artifact paths:

| Paper claim | Source artifact |
|---|---|
| Holdout R² = 0.485 [0.330, 0.604] (Abstract, Table 3, Conclusions) | `eval_outputs/chronological/chronological_regression_ensemble_20220101_20240101.json` |
| Multi-seed bootstrap R² = 0.433 ± 0.092 | `eval_outputs/leak_test_results/baseline_seed{42..46}_predictions.parquet` |
| Paper Table 1 + Table 2 (single-seed paper sample) | `eval_outputs/paper_metrics/{table1_model_comparison,table2_per_site,paper_metrics}.{csv,json}` |
| Spike transition recall (Table 5, §6.3) | `eval_outputs/final_verification/spike_detection_eval.log` |
| Component ablation (Appendix Table A1, §5.5) | `eval_outputs/final_verification/paper_ablation_results_grid_winner.json` |
| Stability perturbation (Appendix Table A2) | `eval_outputs/final_verification/paper_stability_quick.log` |
| OAD ↔ ESP correlations (offshore validation) | `ocean anomaly detection/OAD_STORYLINE.md` Chapter 9 |

The journey from each early-stage number to the current authoritative value is
documented in [`CORRECTED_NUMBERS.md`](CORRECTED_NUMBERS.md).

---

## "How do I verify the headline R² = 0.485 in 3 commands?"

```bash
# 1. Pull the chronological eval result + inspect the headline metric block
.venv/bin/python -c "
import json
d = json.load(open('eval_outputs/chronological/chronological_regression_ensemble_20220101_20240101.json'))
ov = d['overall']
print(f'R²:    {ov[\"r2\"][0]:.3f}  [95% CI: {ov[\"r2\"][1]:.3f}, {ov[\"r2\"][2]:.3f}]')
print(f'MAE:   {ov[\"mae\"][0]:.2f} µg/g')
print(f'N:     {ov[\"n\"]} real DA measurements (spikes: {ov[\"n_spikes\"]})')
print(f'Spike recall: {ov[\"spike_recall\"]:.3f}, F2: {ov[\"spike_f2\"]:.3f}')
"
# Expected output: R²: 0.485 [0.330, 0.604], MAE: 6.76, N: 404

# 2. Run the regression-test suite to confirm the source artifacts haven't drifted
.venv/bin/python -m pytest tests/test_chronological_eval_smoke.py -v
# Expected: 3 passed (smoke tests assert R² is in expected range)

# 3. Verify the leak-free protocol: holdout is downstream of all selection
.venv/bin/python -c "
from forecasting.tuned_config import get_eval_windows
w = get_eval_windows()
print(f'Validation window: [{w[\"validation_start\"]}, {w[\"validation_end\"]}) — used by tuning')
print(f'Holdout window:    [{w[\"holdout_start\"]}, {w[\"holdout_end\"]}) — NEVER inspected during selection')
"
# Expected: Validation [2019-01-01, 2022-01-01), Holdout [2022-01-01, 2024-01-01)
```

---

## "How do I verify there's no data leakage?"

```bash
# 1. Inspect the per-prediction leakage assertion
grep -A 12 "def _verify_no_data_leakage" forecasting/raw_forecast_engine.py
# Expected: an assertion that training_data["date"].max() <= anchor_date

# 2. Run the leakage audit test
.venv/bin/python -m pytest tests/test_leakage_audit.py -v
# Expected: passes (the engine refuses to train on post-anchor data)

# 3. Confirm the 7-day forecast horizon is invariant
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('eval_outputs/holdout_validation/baseline_predictions.parquet')
gaps = (pd.to_datetime(df['date']) - pd.to_datetime(df['anchor_date'])).dt.days
print(f'gap distribution: min={gaps.min()}, max={gaps.max()}, mean={gaps.mean():.1f}')
assert (gaps == 7).all(), 'Forecast horizon contract broken!'
"
# Expected: all 1177 rows have gap=7
```

The grid-search per-site config selection (which defines ensemble weights and
prediction clipping) is leak-free by construction: it uses 3-fold chronological
CV on pre-2022 data only. See `scripts/eval/grid_search_weights_clip.py` lines
21–25 for the fold definition.

---

## "How do I verify any spike-detection claim?"

```bash
# All spike numbers came from one re-run on Hyak (job 35522553, 2026-05-23)
cat eval_outputs/final_verification/spike_detection_eval.log | grep -A 2 "Hypothesis Test"
# Expected output:
#   Naive transition recall:    0.236
#   Ensemble transition recall: 0.145
#   Spike classifier transition recall: 0.327
#   Improvement factor:                 1.4x

# Hybrid alert numbers (classifier OR regression>12) on full panel:
.venv/bin/python -c "
import pandas as pd
from sklearn.metrics import recall_score, precision_score, fbeta_score
df = pd.read_parquet('eval_outputs/holdout_validation/baseline_predictions.parquet')
df['date'] = pd.to_datetime(df['date']); df = df.sort_values(['site','date']).reset_index(drop=True)
trans = (df.groupby('site')['actual_da'].shift(1) < 20) & (df['actual_da'] > 20)
y_true = (df['actual_da']>20).astype(int)
y_alert = df['spike_alert'].fillna(False).astype(int)
print(f'Hybrid event recall:      {recall_score(y_true, y_alert):.3f}')
print(f'Hybrid transition recall: {recall_score(y_true[trans], y_alert[trans]):.3f}')
"
# Expected: event recall 0.876, transition recall 0.803
```

These match the paper Table 5 (Trans. Rec. 0.803, Spike Rec. 0.876).

---

## "How do I re-run the tuning workflow from scratch?"

The entire selection workflow is reproducible end-to-end:

```bash
# On Hyak (4–6 hours total compute, parallel sbatch arrays):
ssh klone-login
cd /gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid
sbatch scripts/tune/tune_all.sbatch

# Orchestrator submits 5 sweep jobs in parallel:
#   1. Per-site grid search (leak-free 3-fold CV on pre-2022)
#   2. Per-site xgb_params perturbation verification
#   3. Global XGB defaults sweep (8 configs)
#   4. MIN_TRAINING_FOR_TUNING sweep (5 cutoff values)
#   5. Spike classifier hyperparameter sweep
# Waits for all to complete, then assembles a new
# config/tuned_hyperparameters.json with refreshed provenance.

# To verify the new JSON matches the prior protocol:
.venv/bin/python -m pytest tests/test_tuned_config.py tests/test_grid_winners_apply.py
```

If the orchestrator's winners differ materially from the committed JSON, the
test suite will fail with a clear diff. Update both the test fixture and the
JSON in the same commit to record the change.

---

## "What's the relationship between code, config, and paper?"

```
config/tuned_hyperparameters.json     ← single source of truth
       │                                (all tunable hyperparameters + per-key provenance)
       ├── consumed by forecasting/tuned_config.py
       │       (with @lru_cache(1) — read once per process)
       │
       ├── consumed by forecasting/per_site_models.py
       │       (SITE_SPECIFIC_CONFIGS = get_per_site())
       │
       ├── consumed by config.py
       │       (SPIKE_CLASSIFIER_PARAMS, PARAM_GRID, ZERO_IMPORTANCE_FEATURES, ...)
       │
       └── tested by tests/test_tuned_config.py + tests/test_grid_winners_apply.py
              (CI runs these on every PR)

paper/datect_paper_mdpi.tex           ← every number traces to eval_outputs/
       ├── Abstract + Table 3 + Conclusions  → eval_outputs/chronological/
       ├── Table 1 + Table 2                 → eval_outputs/paper_metrics/
       ├── Table 5 + §5.5 + §6.3            → eval_outputs/final_verification/
       └── Appendix A1 + A2                  → eval_outputs/final_verification/
```

---

## "Where do I report a discrepancy?"

If you find any paper claim that doesn't match its source artifact:

1. Run the regression tests: `pytest tests/`
2. Check `docs/CORRECTED_NUMBERS.md` for the audit trail of historical
   corrections — your discrepancy may already be documented as a journey step.
3. If still unresolved, open a GitHub issue at
   <https://github.com/ansoncchen/DATect-Forecasting-Domoic-Acid/issues>
   with the specific paper line + the verified value.

---

## File map: where to find what

| Want to inspect | Look here |
|---|---|
| Headline numbers + journey | `docs/CORRECTED_NUMBERS.md` |
| OAD subproject story | `ocean anomaly detection/OAD_STORYLINE.md` |
| DATect-side OAD integration deep dive | `docs/OAD_INTEGRATION_RESULTS.md` |
| Experiment compendium (5 chain experiments + tuning) | `docs/EXPERIMENT_SUMMARY.md` |
| Project conventions + mechanical gotchas | `CLAUDE.md` |
| Reproducible per-site configs | `config/tuned_hyperparameters.json` |
| Methodology deep dive (engine internals) | `docs/PIPELINE_DEEP_DIVE.md` |
| Data pipeline | `docs/DATA_PIPELINE_DETAILED.md` |
| Hyak workflow | `docs/HYAK_SETUP.md` |
| Web dashboard guide | `docs/VISUALIZATIONS_GUIDE.md` |

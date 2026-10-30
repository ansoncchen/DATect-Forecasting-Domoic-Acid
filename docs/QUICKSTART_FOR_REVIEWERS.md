# Quickstart for Reviewers

How to trace each claim in the DATect paper to a committed artifact, check the
leak-free protocol, and re-run the evaluations. The numbers themselves, and
which run each comes from, are in [`CORRECTED_NUMBERS.md`](CORRECTED_NUMBERS.md);
this page does not repeat them.

Run everything from the repository root. Locally, use a virtual environment with
`requirements.txt` and `requirements-dev.txt` installed (the project's own is
`.venv/bin/python`). Heavy runs belong on Hyak (see [`HYAK_SETUP.md`](HYAK_SETUP.md)).

---

## Where each claim comes from

| Paper claim | Artifact |
|---|---|
| Holdout (2022–2023) regression metrics: Abstract, Table 3, Conclusions | `results/chronological/chronological_regression_ensemble_20220101_20240101.{json,parquet}` |
| Validation-window (2019–2021) metrics | `results/chronological/chronological_regression_ensemble_20190101_20220101.{json,parquet}` |
| 5-seed bootstrap confirmation | `results/multi_seed/baseline_seed{42..46}_predictions.parquet` |
| Tables 1–2 and the 4-category classification table (single-seed paper sample) | `results/paper_metrics/{table1_model_comparison,table2_per_site,table3_classification}.csv`, `paper_metrics.json` |
| Spike / transition recall, including the hybrid alert (Table 5, Figure 4) | `results/spike_metrics_seed123/{spike_metrics_by_model,per_site_metrics,threshold_sweep}.csv`, log in `results/final_verification/spike_detection_eval.txt` |
| Environmental lead signals before spikes (exploratory, full data) | `results/env_lead/{spike_events,statistical_tests}.csv`, log in `results/final_verification/spike_env_lead_analysis.txt` (`scripts/eval/spike_env_lead_analysis.py`) |
| Persistence ceiling per site | `results/final_verification/autocorrelation_diagnostic.txt` (`scripts/eval/autocorrelation_diagnostic.py`) |
| Component ablation (Appendix Table A1) | `results/ablation/paper_ablation_results.json`, log in `results/final_verification/paper_ablation_study.txt` |
| Feature-group ablation (appendix) | `results/ablation/paper_features_results.json` (`scripts/eval/paper_sensitivity_study.py --study features`) |
| Design perturbations (Appendix Table A2) | `results/ablation/paper_perturbations_results.json` (`paper_sensitivity_study.py --study perturbations`) |
| Per-site R² by seed (Appendix Table A3) | `results/multi_seed/per_site_r2_by_seed.csv` (`scripts/eval/seed_stability_table.py`) |
| Spike-classifier feature importance | `results/spike_classifier/importance_by_seed.csv` (`scripts/eval/spike_classifier_importance.py`) |
| Per-anchor tuning on vs off; tuned vs baseline configs | `results/holdout_validation/*_predictions.parquet` |
| How each tuned value was chosen | `config/tuned_hyperparameters.json` (`provenance`) and `results/tuning_provenance/` |
| OAD vs ESP mooring correlations | [`../oad/OAD_STORYLINE.md`](../oad/OAD_STORYLINE.md), Chapter 9 |

The committed results come from the October 2026 cleanup re-run at commit
`a2de8dac` (golden baseline below). `results/holdout_validation/` and
`results/tuning_provenance/` predate the cleanup: they document how the per-site
configuration was chosen and are not re-run.

---

## Read the headline metrics

```bash
.venv/bin/python -c "
import json
d = json.load(open('results/chronological/chronological_regression_ensemble_20220101_20240101.json'))
ov = d['overall']
print('window:', d['window'])
print(f'R2  {ov[\"r2\"][0]:.3f}  95% CI [{ov[\"r2\"][1]:.3f}, {ov[\"r2\"][2]:.3f}]')
print(f'MAE {ov[\"mae\"][0]:.2f} ug/g   N={ov[\"n\"]}  spikes={ov[\"n_spikes\"]}')
print(f'spike recall {ov[\"spike_recall\"]:.3f}  precision {ov[\"spike_precision\"]:.3f}  F2 {ov[\"spike_f2\"]:.3f}')
"
```

The 5-seed confirmation is the holdout-window R² of each seed's predictions. The
API computes it the same way (`GET /api/metrics/summary`, which the dashboard shows):

```bash
.venv/bin/python -c "
import glob, numpy as np, pandas as pd
from sklearn.metrics import r2_score
r2 = []
for p in sorted(glob.glob('results/multi_seed/*_predictions.parquet')):
    df = pd.read_parquet(p); df['date'] = pd.to_datetime(df['date'])
    h = df[(df['date'] >= '2022-01-01') & (df['date'] < '2024-01-01')]
    r2.append(r2_score(h['actual_da'], h['predicted_da']))
print(f'{len(r2)} seeds: R2 {np.mean(r2):.3f} +/- {np.std(r2):.3f}')
"
```

`pytest tests/test_chronological_eval_smoke.py` checks that the holdout artifact
covers only its window, that its metrics are in the expected range, and that it
was produced with the grid-winner per-site configuration.

---

## Check the leak-free protocol

```bash
# 1. The per-prediction assertion: training data must end at the anchor date
grep -A 12 "def _verify_no_data_leakage" forecasting/raw_forecast_engine.py

# 2. Leakage audit tests (anchor cut-off, persistence from training data only, past-only lags)
.venv/bin/python -m pytest tests/test_leakage_audit.py -v

# 3. The forecast horizon is exactly 7 days for every holdout prediction
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('results/chronological/chronological_regression_ensemble_20220101_20240101.parquet')
gap = (pd.to_datetime(df['date']) - pd.to_datetime(df['anchor_date'])).dt.days
assert (gap == 7).all(), gap.value_counts()
print('all', len(df), 'rows have a 7-day horizon')
"

# 4. The split: tuning objectives use the validation window; the holdout is never inspected
.venv/bin/python -c "
from forecasting.tuned_config import get_eval_windows
w = get_eval_windows()
print('validation', w['validation_start'], '->', w['validation_end'])
print('holdout   ', w['holdout_start'], '->', w['holdout_end'])
"
```

The per-site grid search that sets ensemble weights and clipping uses 3-fold
chronological CV on pre-2022 data only (folds defined in the docstring of
`scripts/eval/grid_search_weights_clip.py`). OAD features carry an extra 5-day
lag so the centered MODIS composites never overlap the forecast window
(`forecasting/oad_features.py`, [`OAD_INTEGRATION_RESULTS.md`](OAD_INTEGRATION_RESULTS.md)).

---

## Check a spike-detection claim

```bash
grep -A 12 "Hypothesis Test" results/final_verification/spike_detection_eval.txt
```

The hybrid alert (spike classifier OR regression forecast ≥ 12 µg/g) is stored per
prediction as `spike_alert`. To score it with the project's spike rule (actual
≥ 20 µg/g, shared through `forecasting/metrics.py`) on any predictions parquet:

```bash
.venv/bin/python -c "
import sys, pandas as pd
from forecasting.metrics import exceeds, detection_scores
df = pd.read_parquet(sys.argv[1])
df['date'] = pd.to_datetime(df['date']); df = df.sort_values(['site', 'date']).reset_index(drop=True)
spike = exceeds(df['actual_da'], 20.0)
trans = (df.groupby('site')['actual_da'].shift(1) < 20.0).to_numpy() & spike
alert = df['spike_alert'].fillna(False).astype(bool).to_numpy()
print('spike events:', detection_scores(spike, alert))
print('transition recall:', detection_scores(spike[trans], alert[trans])['recall'])
" results/chronological/chronological_regression_ensemble_20220101_20240101.parquet
```

A transition is a spike whose previous measurement at the same site was below 20 µg/g.

---

## Re-run every evaluation (golden baseline)

`scripts/eval/golden_baseline.sh` runs every evaluation behind a reported number
for one git ref, in a disposable worktree on Hyak scrubbed storage, as one Slurm job:
chronological validation and holdout, multi-seed 42–46, paper metrics (seed 123,
40% sample) and spike detection.

```bash
ssh klone-login
cd /gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid      # any Hyak checkout
bash scripts/eval/golden_baseline.sh <git-ref> [label]
# outputs: ${GOLDEN_ROOT:-/gscratch/scrubbed/$USER/golden}/<label>

# Compare two runs: predictions must match exactly, metric tables within 1e-12
python scripts/eval/compare_golden.py <golden_root>/<label_a> <golden_root>/<label_b>
```

`compare_golden.py` exits 0 when the runs are identical and otherwise prints the
headline-metric deltas for each file that differs. The job uses
`/gscratch/stf/ac283/envs/datect_scratch/bin/python` unless `DATECT_PY` is set.

Single evaluations, from the repo root:

```bash
python scripts/eval/chronological_eval.py --window 2022-01-01:2024-01-01   # default window: holdout
sbatch scripts/eval/multi_seed_baseline.sbatch                            # array 0-4 = seeds 42-46
python scripts/eval/eval_paper_metrics.py --seed 123 --force-rerun
python scripts/eval/spike_detection_eval.py --seed 123 --sample-fraction 0.40
```

---

## Re-run the tuning

```bash
# On Hyak, from the repo root (sbatch jobs run the checkout they were submitted from)
sbatch scripts/tune/tune_all.sbatch
```

`tune_all.py` submits four sweeps and waits for them, then writes a new
`config/tuned_hyperparameters.json` with provenance:

1. Per-site grid search over ensemble weights and clipping (leak-free 3-fold chronological CV on pre-2022)
2. Per-site XGBoost parameter perturbation check
3. Global XGBoost defaults sweep (8 configs)
4. Spike-classifier sweep (6 configs)

`--dry-run` prints what would happen; `--assemble-only` rebuilds the JSON from
existing sweep outputs. Afterwards run
`pytest tests/test_tuned_config.py tests/test_grid_winners_apply.py`: the second
file pins the current grid winners, so new winners must update the JSON and that
fixture in the same commit.

---

## Script index

Run from the repo root; `scripts/eval/_repo.py` puts the repo root on `sys.path`
and changes into it. None of these are imported by the API or `run_datect.py`.

**`scripts/eval/`**

| Script | Purpose | Default output |
|---|---|---|
| `chronological_eval.py` | Deterministic evaluation on every real DA measurement in a date window, with row-bootstrap CIs | `results/chronological/` |
| `multi_seed_baseline.py` (+ `.sbatch`) | One sampled retrospective per seed (array 0–4 = seeds 42–46) | `results/multi_seed/` |
| `eval_paper_metrics.py` | Paper Tables 1–3: model comparison, per-site metrics, 4-category classification, bootstrap CIs | `results/paper_metrics/` |
| `spike_detection_eval.py` | Spike and transition detection of each model vs naive persistence | `results/spike_metrics[_seed<N>]/` |
| `paper_ablation_study.py` | Component ablations (interpolated training, per-site config, lags, derived features, OAD), each in a fresh subprocess | `results/ablation/paper_ablation_results.json` |
| `grid_search_weights_clip.py` (+ `.sbatch`) | Per-site grid over ensemble weights × clip quantile × clip max, chronological 3-fold CV | `results/tuning_provenance/grid_search_results/` |
| `validate_tuned_on_holdout.py` | Runs a labeled config, then `--compare A B` scores both by window and prints a REAL / OVERFITTING / NEUTRAL verdict | `results/holdout_validation/` |
| `golden_baseline.sh` / `.sbatch` | Re-run every reported evaluation for a git ref (Hyak) | worktree under `/gscratch/scrubbed/$USER/golden/` |
| `compare_golden.py` | Diff two golden runs | stdout |
| `autocorrelation_diagnostic.py` | Per-site lag-1 autocorrelation; ρ² bounds a persistence-only forecast | stdout |
| `quick_raw_retrospective_compare.py` | Fast naive vs Ridge vs ensemble comparison on a small sample of raw DA | stdout / `--output-json` |
| `tiny_sequence_baseline.py` | sklearn MLP on the last K weekly feature windows, same leak-free rows | stdout |
| `run_per_site_xgb_verify.sbatch`, `run_global_xgb_sweep.sbatch`, `run_spike_classifier_sweep.sbatch` | Tuning stages 2–4 | `results/tuning_provenance/` |
| `_repo.py`, `_subprocess.py` | Helpers: repo-root setup; run engine code in a fresh interpreter so `DATECT_*` variables take effect | — |

**`scripts/tune/`**: `tune_all.py` (+ `tune_all.sbatch`), the re-tuning orchestrator above.

**`scripts/data/`**

| Script | Purpose |
|---|---|
| `rebuild_final_output.py --recompute {measurements,oad,all} [--check]` | Rebuild `data/processed/final_output.parquet` offline: recompute `da`/`pn` and/or the `oad_*` columns, keep every downloaded column unchanged |
| `export_oad_features.py <checkpoint>` | Validate one OAD checkpoint's regional scores and write `data/processed/oad_scores.parquet` |

---

## How code, config, results and paper connect

```
config/tuned_hyperparameters.json        all tuned values + per-key provenance
   ├── forecasting/tuned_config.py       loads it once per process
   ├── forecasting/per_site_models.py    SITE_SPECIFIC_CONFIGS
   ├── config.py                         global model params, spike thresholds, eval windows
   └── tests/test_tuned_config.py, tests/test_grid_winners_apply.py

papers/datect/datect_paper_mdpi.tex
   ├── Abstract, Table 3, Conclusions    results/chronological/
   ├── Tables 1-2, classification table  results/paper_metrics/
   ├── Table 5                           results/final_verification/spike_detection_eval.txt
   └── Appendix A1, A2                   results/ablation/, results/final_verification/
```

---

## Reporting a discrepancy

1. Run the tests: `pytest tests/`.
2. Check [`CORRECTED_NUMBERS.md`](CORRECTED_NUMBERS.md); the difference may already be recorded there.
3. Otherwise open an issue at
   <https://github.com/ansoncchen/DATect-Forecasting-Domoic-Acid/issues>
   with the paper line and the value you found.

## File map

| Want to inspect | Look here |
|---|---|
| Reported numbers and their history | `docs/CORRECTED_NUMBERS.md` |
| Tuned per-site configuration | `config/tuned_hyperparameters.json` |
| Engine internals and safeguards | `docs/PIPELINE_DEEP_DIVE.md` |
| Data pipeline | `docs/DATA_PIPELINE_DETAILED.md` |
| DATect-side OAD integration | `docs/OAD_INTEGRATION_RESULTS.md` |
| OAD subproject | `oad/OAD_STORYLINE.md`, `oad/RESULTS.md` |
| Project conventions and gotchas | `CLAUDE.md` |
| Hyak workflow | `docs/HYAK_SETUP.md` |
| Dashboard charts | `docs/VISUALIZATIONS_GUIDE.md` |

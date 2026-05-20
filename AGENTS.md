## Learned User Preferences

- Keep **gap-fill / synthetic training targets** (what fills sparse DA between samples) separate from **forecast model choice** (XGB, RF, MLP, etc.); they are different experiments and are easy to mix up.
- When comparing alternatives to **causal exponential decay** gap-fill, treat **leak-free, past-only** training rules as non-negotiable; bidirectional imputers are not drop-in replacements without an explicit leakage story.
- Judge whether a model beats the current stack using the same **leak-free raw retrospective** setup as the main engine, not metrics on dense ISO-week **panel** imputation quality.
- Ask for **simple explanations** when the thread moves between synthetic-data design and prediction-model ablations.
- Run all heavy compute on Hyak (`/gscratch/stf/ac283/...`), not locally. Local laptop is for editing code and viewing figures; SSH ControlMaster `klone-login` is set up.

## Learned Workspace Facts

- **Retrospective / paper eval scripts** live under `scripts/eval/`:
  - `eval_paper_metrics.py` — main retrospective R²/MAE/RMSE/classification metrics with bootstrap CIs; flags `--seed`, `--sample-fraction`, `--force-rerun`, `--output-dir`, `--temporal-holdout`. This is the script to run for any "with feature X vs without feature X" comparison (run it twice with different dataset versions).
  - `paper_ablation_study.py` — runs 4 hard-coded ablations as subprocesses with config env-var overrides (no_interpolated_training, no_per_site_models, no_lag_features, no_derived_features). Pattern to follow for adding new ablations.
  - `paper_stability_study.py` + `paper_stability_table.py` — multi-seed / perturbation stability analysis.
  - `spike_detection_eval.py` — spike-event-specific evaluation.
  - There is **NO** `scripts/eval/quick_raw_retrospective_compare.py` — earlier notes mentioning it were stale; the actual entry points are above.
- Typical raw DA volume is **thousands** of shore rows and **thousands** of site-weeks with real measurements; the weekly panel is dense with gap-filled `da`, while evaluation for forecasting skill should track **raw** measurements.
- Quick retrospective and small **MLP/sklearn**-style baselines at modest sample fractions are **CPU-viable** on Hyak; a GPU is not required for that evaluation tier.
- **Ocean anomaly detection subproject** lives at `ocean anomaly detection/` (branch `ocean-anomaly-v2`). It produces per-region regional ocean-state anomaly scores from a 22-year MODIS Aqua cube. The best checkpoint is `ae_3d_l32_c4_t4_s42_mae030` (Phase C MAE-trained 3D ConvAE). Scores live in `ocean anomaly detection/outputs/scores/*.parquet`. See `ocean anomaly detection/RESULTS.md` for headline numbers. Integration into DATect's per-site forecast (adding `oad_*` features) is a planned follow-up.
- **MODIS 8-day composites on ERDDAP are CENTERED on the labeled date** (`long_name = "Centered Time"`). So a score at date *t* contains ~3-4 days of data from after *t*. Any leakage-safe consumer must lag at least 4 more days beyond DATect's existing `anchor = test_date − 7` convention — recommend `test_date − 12` for OAD score lookup.

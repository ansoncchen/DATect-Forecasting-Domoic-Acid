# DATect / OAD experiment compendium

Single-page index of every experiment run on the `oad-integration` branch, with
its result and the final disposition. Cross-references the deep-dive in
`docs/OAD_INTEGRATION_RESULTS.md` (sections noted in brackets).

## Baselines (the numbers to compare against)

**Audited 2026-05-23. All values below are 5-seed (42-46) mean ± std on current `per_site_models.py`. See `docs/CORRECTED_NUMBERS.md`.**

| Metric | Value | Window | N (per seed) | Notes |
|---|---:|---|---:|---|
| Ensemble R² | **0.386 ± 0.145** | 2022-2024 holdout | ~160 | single-seed range 0.19–0.60 |
| Ensemble R² | 0.377 ± 0.164 | 2019-2022 validation | ~220 | Optuna tuning window |
| Ensemble R² | 0.316 ± 0.079 | all-years pooled | ~1190 | tighter spread, more samples |
| MAE | 6.03 ± 0.63 µg/g | 2022-2024 holdout | ~160 | more stable than R² |
| Spike F2 (regression-only) | 0.738 ± 0.024 | 2019-2022 validation | ~220 | most stable spike metric |
| Spike recall (regression-only) | 0.848 ± 0.044 | 2022-2024 holdout | ~160 | most defensible operational number |
| Hybrid alert recall | 0.876 | pooled (seed 123) | 1177 | classifier OR regression union |

**The previously-quoted "R² = 0.492 holdout" was a lucky seed (top of five). Honest paper number is 0.39 ± 0.15.** Sources: `multi_seed_results/baseline_seed{42..46}_predictions.parquet`, generated via `scripts/eval/multi_seed_baseline.py`.

## Experiments — all null or marginal

| # | Experiment | ΔR² (pooled) | Best per-site | Disposition |
|---|---|---:|---|---|
| 1 | **OAD features** (16 cols, mae050) | ≈0 | SW WA slightly + | KEPT in per_site_models.py (per "unless clearly harmful" rule); §3 |
| 2 | **OAD on small-N sites** | mixed | — | not promoted; §5 |
| 3 | Optuna 18-dim full search | val +0.07 / holdout −0.16 | catastrophic on holdout | DISCARDED; §18 |
| 4 | Spike classifier tuning | val F2 +0.05 | holdout pending | held; §18 |
| 5 | Chain 1: `lagged_pn` | +0.0003 | Long Beach +0.007 | within noise; §19 |
| 6 | Chain 2: `beuti_derivatives` | −0.0012 | none | §19 |
| 7 | Chain 3: `nemo_mooring` | −0.0004 | none | §19 |
| 8 | Chain 4: `esp_offshore_pda` | −0.0002 | none | §19 |
| 9 | Chain 5: `ndbc_wind` | −0.0025 | none | §19 |
| 10 | Grid search (w_xgb × clip_q × clip_max) | mostly NaN/negative | Copalis +0.44 (fold-CV, unverified) | NOT applied — no holdout val; §20 |

## The two findings worth keeping

1. **OAD validates at the offshore source** (ESP mooring: Pn r=+0.46, pDA r=+0.33,
   bootstrap CIs exclude zero) but **null at the beach**. The 24 km transport +
   1–2 wk razor clam bioaccumulation chain destroys the predictive signal.
   This is the scientifically interesting result for the poster/writeup. §16

2. **DATect's existing feature set + hand-tuned per-site config is at a flat
   local optimum.** 5 chain experiments, 18-dim Optuna tuning, and 75-config
   grid search per site all failed to move pooled R² by more than seed noise.
   Future gains require new in-situ data (ORHAB-style PN counts at the other
   9 sites), not more derived features from existing parquets. §17

## Files

- `forecasting/oad_features.py` — 16-feature builder + tests
- `forecasting/per_site_models.py` — env-var hooks (HPARAM_OVERRIDE_JSON, OAD_ON_SMALL_N)
- `chains/c{1..6}_*.py` + `chains/run_chain.py` + `chains/run_chain.sbatch` — chain experiments
- `scripts/eval/grid_search_weights_clip.py` — constrained grid search
- `scripts/eval/multi_seed_baseline.py` — 5-seed bootstrap
- `scripts/eval/validate_tuned_on_holdout.py` — REAL/OVERFITTING/NEUTRAL/NULL verdict
- `docs/OAD_INTEGRATION_RESULTS.md` — full writeup
- `poster_figures/` — Canva-ready PNGs (study region + architecture)

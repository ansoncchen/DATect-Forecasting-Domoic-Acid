# DATect / OAD experiment compendium

Single-page index of every experiment run on the `oad-integration` branch, with
its result and the final disposition. Cross-references the deep-dive in
`docs/OAD_INTEGRATION_RESULTS.md` (sections noted in brackets).

## Baselines (the numbers to compare against)

| Metric | Value | Window | N | Source |
|---|---:|---|---:|---|
| Ensemble R² | **0.492** | 2022-2024 holdout | 164 | retro eval, current `per_site_models.py` |
| Ensemble R² | 0.346 | 2019-2022 validation | 216 | same |
| Ensemble R² | 0.173 | random-anchor pooled, seed 123 | 1202 | seed-sensitive, ±0.13 across seeds 42-46 |
| MAE | 5.33 µg/g | 2022-2024 holdout | 164 | more stable than R² |
| Spike F2 | 0.732 | 2019-2022 validation | 32 spikes | tuned classifier |
| Hybrid alert recall | 0.859 | rolling | — | classifier + regression union |

Multi-seed CIs in `scripts/eval/multi_seed_baseline.py`.

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

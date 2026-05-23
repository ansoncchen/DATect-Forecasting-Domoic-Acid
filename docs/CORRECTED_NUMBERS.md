# Authoritative Numbers — Multi-Seed Audit (2026-05-23)

Re-verified on current `oad-integration` HEAD (`188e05a6`) using `multi_seed_results/baseline_seed{42..46}_predictions.parquet`. **Five independent random-anchor seeds**, current `per_site_models.py`, current `data/processed/final_output.parquet`.

## ⚠️ Headline corrections vs CLAUDE.md and the paper

| Metric | CLAUDE.md / paper (stale, single-seed) | **Authoritative (multi-seed, current code)** |
|---|---|---|
| Pooled all-years R² | 0.173 / 0.215 | **0.316 ± 0.079** (range 0.220 – 0.415) |
| Holdout 2022-2023 R² | **0.492** ← lucky seed | **0.386 ± 0.145** (range 0.188 – 0.599) |
| Holdout MAE (µg/g) | 5.33 | **6.03 ± 0.63** |
| Holdout spike recall | 0.85 (referenced) | **0.848 ± 0.044** ✓ |
| Holdout spike F2 | — | **0.648 ± 0.044** |
| Validation R² | 0.346 | **0.377 ± 0.164** |
| Validation spike F2 | 0.732 | **0.738 ± 0.024** ✓ |
| Hybrid alert recall (rolling) | 0.859 | **0.876** (pooled, single seed 123) |
| Paper temporal-holdout R² | 0.315 (paper) | **0.386 ± 0.145** (post-OAD branch) |

**The "0.492 holdout R²" that's been a quoted headline was the highest seed of five.** Honest paper number is **R² ≈ 0.39 with 5-seed range 0.19–0.60.**

## Per-site holdout 2022-2023 (5-seed mean)

| Site | R² mean ± std | R² range | MAE mean | N (mean per seed) |
|---|---|---|---|---|
| Coos Bay | **+0.70 ± 0.09** | +0.56 to +0.82 | 11.8 | 10 ← tiny N, lucky |
| Quinault | **+0.56 ± 0.18** | +0.35 to +0.82 | 3.0 | 26 |
| Long Beach | **+0.55 ± 0.17** | +0.30 to +0.81 | 3.9 | 20 |
| Twin Harbors | **+0.53 ± 0.17** | +0.37 to +0.85 | 4.1 | 18 |
| Copalis | **+0.52 ± 0.16** | +0.29 to +0.71 | 3.4 | 29 |
| Clatsop Beach | **+0.50 ± 0.20** | +0.30 to +0.88 | 6.0 | 12 |
| Kalaloch | **+0.42 ± 0.17** | +0.21 to +0.68 | 2.8 | 9 |
| Newport | **−1.06 ± 1.70** | −4.40 to +0.38 | 16.9 | 20 |
| Gold Beach | **−2.14 ± 3.21** | −8.46 to −0.08 | 5.8 | 17 |
| Cannon Beach | (no spikes in holdout) | — | — | <5 |

**5 WA sites: mean R² = 0.50, std 0.17, all reliably positive.** **3 OR sites with measurable performance: 1 positive (Clatsop), 2 deeply negative (Newport, Gold Beach).** Coos Bay's +0.70 is misleading — only ~10 holdout samples per seed.

## What's confirmed unchanged

- **OAD is null at the beach** (`no_oad_features` ΔR² = +0.0015 in `paper_ablation_results.json` seed 123 — i.e. dropping OAD slightly *helps*, within noise).
- **Per-site customization is the biggest lever** (ΔR² = −0.153 without it, seed 123).
- **5 chain experiments all null** (|ΔR²| < 0.003 pooled; lagged_pn helps Long Beach +0.007 within seed noise).
- **OAD ↔ ESP correlations remain valid** (r=+0.46 Pn, r=+0.33 pDA — these come from a separate analysis not affected by per_site_models state).

## What the paper must change

The current `paper/datect_paper_mdpi.tex` (line 62 abstract, line 406 Table 1, line 514 per-site discussion, line 516 temporal-holdout pooled R²) reports **single-seed** point estimates without bootstrap CIs across seeds. Required edits:

1. **Abstract (line 62)**: replace "All five Washington sites achieve high skill ($R^2 = 0.480$–$0.789$)" with multi-seed range: "All five Washington sites achieve high skill ($R^2$ mean 0.42–0.56 across 5 random-anchor seeds, range 0.21–0.85)."
2. **Table 1 (line 409)**: regenerate with multi-seed mean ± std, or add explicit "seed = 123" qualifier and a footnote pointing to a new "Multi-seed sensitivity" appendix.
3. **Table 2 per-site (line 435)**: use the per-site multi-seed table above. Specifically, **Coos Bay's R² = 0.616 in the paper is at single-seed N=89; multi-seed mean is +0.70 ± 0.09 but only N≈10 per seed at holdout.** Wording needs to convey this is small-N-driven.
4. **Temporal holdout (line 463)**: pooled R² = 0.315 → 0.386 ± 0.145; WA mean 0.549 → 0.50 ± 0.17; OR mean 0.113 → mixed (1 positive, 2 deeply negative).
5. **Transition recall section (line 550)**: re-verify the 12.4% regression / 23.6% naive / 73.4% classifier numbers. My local re-computation with the current `spike_alert` definition gives different values (0.84 / 0.84 / 0.81), suggesting either a definitional change since the paper or that the paper used a stricter "transition" definition (e.g. previous DA < 5, not just < 20). Need to read the paper's exact event-definition code.
6. **New paragraph in Methods / Limitations**: explicitly call out single-seed-to-multi-seed sensitivity. The paper currently mentions a "five-seed perturbation study" (line 377) but doesn't quote its dispersion.

## Final Hyak job results (completed 2026-05-23)

### Job 35510043 — paper-metrics N=2181 (40% sample, seed 123)
From `eval_results/paper_metrics/`:

| Model | R² [95% CI] | MAE [95% CI] | RMSE |
|---|---|---|---|
| **Ensemble** | **0.241 [0.125, 0.364]** | **6.51 [5.89, 7.22]** | 17.50 |
| XGBoost | 0.276 [0.185, 0.379] | 6.58 [5.90, 7.25] | 17.08 |
| Random Forest | 0.195 [0.060, 0.324] | 6.46 [5.79, 7.23] | 18.02 |
| Linear/Ridge | 0.203 [0.072, 0.326] | 6.61 [5.93, 7.39] | 17.93 |
| Naïve persistence | −0.426 [−0.951, −0.034] | 7.73 [6.84, 8.76] | 23.98 |

Per-site (paper Table 2 format):

| Site | N | R² [95% CI] | MAE [95% CI] | RMSE | 4-cat acc |
|---|---:|---|---|---:|---:|
| Copalis | 277 | 0.789 [0.706, 0.843] | 2.82 [2.31, 3.39] | 5.28 | 0.83 |
| Long Beach | 209 | 0.629 [0.551, 0.713] | 4.44 [3.47, 5.48] | 8.67 | 0.73 |
| Twin Harbors | 225 | 0.591 [0.469, 0.759] | 4.67 [3.35, 6.21] | 12.25 | 0.81 |
| Quinault | 181 | 0.580 [0.495, 0.733] | 4.16 [2.97, 5.62] | 10.18 | 0.76 |
| Kalaloch | 235 | 0.480 [0.348, 0.641] | 3.77 [2.72, 5.08] | 10.14 | 0.81 |
| Clatsop Beach | 354 | 0.305 [−0.183, 0.558] | 6.12 [4.83, 7.52] | 13.91 | 0.71 |
| Gold Beach | 257 | 0.079 [−0.071, 0.231] | 8.24 [5.83, 11.09] | 23.04 | 0.77 |
| Cannon Beach | 116 | −0.044 [−0.090, −0.009] | 3.91 [1.40, 6.97] | 16.14 | 0.92 |
| Coos Bay | 109 | −0.024 [−0.654, 0.313] | 24.35 [18.55, 30.33] | 39.84 | 0.49 |
| Newport | 218 | −0.135 [−0.986, 0.056] | 11.07 [8.29, 14.79] | 27.03 | 0.54 |

### Job 35510066 — leak test (5 seeds × grid-winner overrides)

| Window | Hand-tuned (current) | Grid-winners (leak-free) | Δ |
|---|---|---|---|
| Pre-2019 (sanity) | 0.273 ± 0.10 | 0.273 ± 0.10 | ±0.00 ± 0.03 ✓ |
| Val 2019-2022 | 0.377 ± 0.16 | 0.315 ± 0.12 | −0.06 ± 0.07 (hand leak) |
| **Holdout 2022-2023** | **0.386 ± 0.15** | **0.434 ± 0.12** | **+0.05 ± 0.07** |
| All anchors | 0.316 ± 0.08 | 0.307 ± 0.08 | −0.01 ± 0.02 |

**Verdict**: Hand-tuning leaked ~0.06 R² on val but NOT on holdout. Grid winners give equivalent or slightly better unbiased performance. Both configurations are statistically indistinguishable on the holdout (Δ within 1σ). Per-site, grid flips 3 sites from RF to XGB (Copalis, Quinault, Twin Harbors).

**Promotion decision**: do NOT promote grid winners. The +0.05 holdout gain is within seed noise; the val degradation is a wash; the change adds complexity for no clean win. Document the leak finding instead — paper should report the unbiased holdout number (0.39 ± 0.15) and acknowledge that the per-site config was tuned with implicit val-window access.

## Files to update after Hyak finishes

1. `CLAUDE.md` — Success Metrics table (replace single-seed values with multi-seed mean ± std)
2. `EXPERIMENT_SUMMARY.md` — same
3. `docs/OAD_INTEGRATION_RESULTS.md` — §3.4 holdout numbers
4. `paper/datect_paper_mdpi.tex` — abstract + Tables 1, 2, 3 + transition recall paragraph
5. `paper/generate_figures.py` — rerun with new artifacts

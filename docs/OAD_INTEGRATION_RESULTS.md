# OAD Integration — Results & Paper-Ready Writeup

> Status: **template** — populate the `{{...}}` placeholders from `paper_ablation_results.json` and the holdout-validation outputs after each Hyak run finishes. The structure here mirrors what should go into the paper; each section has a markdown table that's directly portable to LaTeX `\begin{tabular}` via `pandoc` or manual conversion.

**Branch:** [`oad-integration`](https://github.com/ansoncchen/DATect-Forecasting-Domoic-Acid/tree/oad-integration)
**Last updated:** {{DATE}}
**Reporting model:** `AE_3d_l32_c4_t4_s42_mae050` (3D masked-autoencoder, mask ratio 0.50)

---

## 1. Headline number (for the abstract / introduction)

> *Add to abstract once filled:*
>
> "Augmenting the per-site DA forecasting ensemble with a learned 16-dimensional regional ocean-anomaly representation produced from an unsupervised 3D masked autoencoder on 22 years of MODIS Aqua imagery yields a pooled ensemble R² of {{R2_WITH_OAD}} on the temporal holdout (2022-2024), compared to {{R2_BASELINE}} for the per-site environmental baseline (Δ R² = {{DELTA_R2_POOLED}}; Δ MAE = {{DELTA_MAE_POOLED}} µg/g; N = {{N_HOLDOUT}} test points)."

---

## 2. Methods section additions

### 2.1 OAD feature derivation

Sixteen new features per (site, weekly anchor date R) row, joined from `data/processed/oad_scores.parquet` and `data/processed/oad_cloud_fractions.parquet`:

| Feature group | Count | Per region | Description |
|---------------|-------|-----------|-------------|
| Anomaly score (local + envelope) | 14 | × 2 regions | 7-day mean ending at R−5, plus 7-day lag-1week and lag-2week, plus 30-day mean/max/trend, plus DOY z-score with leave-prior-years climatology |
| Cloud fraction (local + envelope) | 2 | × 2 regions | 7-day mean of valid-pixel-fraction ending at R−5 |

Local region = `SITE_TO_REGION[site]` (one of: Olympic Coast WA, SW Washington / Long Beach, Central Oregon, Southern OR / N CA). Envelope = "Overall (WA–OR–N. CA coastal)" coastwide bounding box.

### 2.2 Leakage analysis (verbatim from the plan)

DATect rows are weekly (Monday). Engine fetches row with date ≤ `anchor_date = test_date − 7` via `get_site_anchor_row()`. MODIS 8-day composite at date D covers `[D − 4, D + 4]`. To guarantee composite-end < anchor:

| Step | Constraint | Value |
|------|-----------|-------|
| Forecast horizon | `(anchor_date, test_date]` | `(test_date − 7, test_date]` |
| Anchor row R | `R ≤ anchor_date` | `R ∈ [test_date − 13, test_date − 7]` |
| OAD feature anchor day D | `D = R − 5` (pre-shift) | `D ∈ [test_date − 18, test_date − 12]` |
| MODIS 8-day composite at D | `[D − 4, D + 4]` | latest day = `D + 4 ∈ [test_date − 14, test_date − 8]` |
| Buffer | (latest composite day) vs `anchor_date` | **1-day margin minimum** |

**Worst case:** when `R = test_date − 7`, latest MODIS data contributing to any OAD feature is `test_date − 8`. Forecast horizon begins at `test_date − 6`. No overlap.

### 2.3 Train / Validation / Test split

```
2003 ──── 2008 ─────────── 2018 ── 2019 ──── 2021 ── 2022 ──── 2024
         |  pre-2019 (training context, not scored)         |
                                  │ VALIDATION │  HOLDOUT   │
                                  │ 2019-2022  │  2022-2024 │
                                   ─────┬─────  ─────┬──────
                                        │             │
                                  Optuna objective    Untouched
                                                      final test
```

- **Training:** all data with `date ≤ anchor_date` (engine-enforced, ~21 years rolling).
- **Validation:** retrospective test points in `[2019-01-01, 2022-01-01)`. Used as the objective for any hyperparameter selection.
- **Holdout:** retrospective test points in `[2022-01-01, 2024-12-31]`. Never seen by any selection process. All reported deltas use this window unless noted.

Pre-2019 retrospective points are NOT scored — they cover the 2014-16 marine heatwave and 2015 PN bloom which would dominate seed-based sampling. The validation window contains the 2019 PN bloom and post-bloom recovery.

---

## 3. A/B results — Task 10 (OAD vs no-OAD)

### 3.1 Pooled metrics (random-anchor sampling, all sites, all years 2008-2024)

Source: `paper_ablation_results.json` (Hyak job 35390792).

| Configuration | R² | MAE (µg/g) | RMSE | N |
|---|---:|---:|---:|---:|
| Baseline (DATect + per-site customization + OAD off) | {{BASELINE_POOLED_R2}} | {{BASELINE_POOLED_MAE}} | {{BASELINE_POOLED_RMSE}} | {{BASELINE_POOLED_N}} |
| **+ OAD (16 features)** | {{TUNED_POOLED_R2}} | {{TUNED_POOLED_MAE}} | {{TUNED_POOLED_RMSE}} | {{TUNED_POOLED_N}} |
| **Δ (+OAD − baseline)** | **{{DELTA_R2_POOLED}}** | **{{DELTA_MAE_POOLED}}** | {{DELTA_RMSE_POOLED}} | — |

### 3.2 Per-site R² (ensemble, all years)

| Site | Region | Baseline R² | +OAD R² | Δ R² | N |
|---|---|---:|---:|---:|---:|
| {{Site1}} | {{Region1}} | {{B1}} | {{W1}} | {{D1}} | {{N1}} |
| ... | ... | ... | ... | ... | ... |

(Per-site table will be auto-populated from `paper_ablation_results.json["per_site"]` after the analyzer runs.)

### 3.3 SW Washington subset (OAD's headline region)

OAD's RESULTS.md headline win was in SW Washington / Long Beach (R² = +0.87, CIΔ vs PCA = [+0.92, +1.05] at lead=1). At the integration-relevant 12-day lead, the AE retains R² ≈ 0.15 in SW WA (mae050 variant). The expected pattern is gain concentrated in:

| SW WA site | Baseline R² | +OAD R² | Δ R² |
|---|---:|---:|---:|
| Twin Harbors | {{TH_B}} | {{TH_W}} | {{TH_D}} |
| Long Beach | {{LB_B}} | {{LB_W}} | {{LB_D}} |
| Clatsop Beach | {{CB_B}} | {{CB_W}} | {{CB_D}} |
| Cannon Beach | {{CnB_B}} | {{CnB_W}} | {{CnB_D}} |

Mean Δ R² in SW WA: **{{SW_WA_MEAN_DELTA}}**.

### 3.4 Holdout-only metrics (2022-2024, the unbiased numbers)

| Configuration | R² (holdout) | MAE (holdout) | N |
|---|---:|---:|---:|
| Baseline (no OAD) | {{HOLDOUT_BASELINE_R2}} | {{HOLDOUT_BASELINE_MAE}} | {{HOLDOUT_N}} |
| **+ OAD** | {{HOLDOUT_TUNED_R2}} | {{HOLDOUT_TUNED_MAE}} | {{HOLDOUT_N}} |
| **Δ** | **{{HOLDOUT_DELTA_R2}}** | **{{HOLDOUT_DELTA_MAE}}** | — |

---

## 4. Ablation context (Task 10 full table)

These rows come from the same `paper_ablation_results.json`. Useful as a calibration: how does the +OAD effect size compare to other architectural choices DATect makes?

| Configuration | R² | Δ R² vs baseline | MAE | N |
|---|---:|---:|---:|---:|
| Baseline (full DATect + OAD) | {{ABL_BASE_R2}} | — | {{ABL_BASE_MAE}} | {{ABL_BASE_N}} |
| No interpolated training | {{ABL_NI_R2}} | {{ABL_NI_DELTA}} | {{ABL_NI_MAE}} | {{ABL_NI_N}} |
| No per-site customization | {{ABL_NP_R2}} | {{ABL_NP_DELTA}} | {{ABL_NP_MAE}} | {{ABL_NP_N}} |
| No observation-order lags | {{ABL_NL_R2}} | {{ABL_NL_DELTA}} | {{ABL_NL_MAE}} | {{ABL_NL_N}} |
| No derived features (pn_log) | {{ABL_ND_R2}} | {{ABL_ND_DELTA}} | {{ABL_ND_MAE}} | {{ABL_ND_N}} |
| **No OAD features (A/B for §3)** | {{ABL_NO_R2}} | {{ABL_NO_DELTA}} | {{ABL_NO_MAE}} | {{ABL_NO_N}} |

---

## 5. Task 11 — Small-N "synthetic data" experiment *(if run)*

Hypothesis: at Coos Bay (N=67), Cannon Beach (N=61), Gold Beach (N=144), and Newport (N=142), OAD's regional signal may compensate for sparse local DA observations.

| Site | N | Baseline R² (no OAD in subset) | +OAD on small-N R² | Δ R² | Verdict |
|---|---:|---:|---:|---:|---|
| Coos Bay | 67 | {{CN_B}} | {{CN_W}} | {{CN_D}} | {{CN_V}} |
| Cannon Beach | 61 | {{CnB_B}} | {{CnB_W}} | {{CnB_D}} | {{CnB_V}} |
| Gold Beach | 144 | {{GB_B}} | {{GB_W}} | {{GB_D}} | {{GB_V}} |
| Newport | 142 | {{NP_B}} | {{NP_W}} | {{NP_D}} | {{NP_V}} |

**Decision rule from [`analyze_oad_ablation.py`](../scripts/eval/analyze_oad_ablation.py):**
- PROMOTE if wins ≥ 3 of 4 AND mean Δ R² > +0.01 → add OAD to all 10 sites
- KEEP SELECTIVE if wins ≤ 1 OR mean Δ R² < −0.01 → handcrafted minimal subsets win
- MIXED otherwise → keep current 5-site selective inclusion

**Verdict:** {{SMALL_N_VERDICT}}

---

## 6. Task 12 — Per-site hyperparameter tuning *(if run)*

Optuna TPE, 30 trials × 10 sites, 18 hyperparameters per site. Objective: validation R² on 2019-2022 retrospective points. Final verdict from holdout (2022-2024).

### 6.1 Window-level summary

| Window | Baseline R² | Tuned R² | Δ R² | N |
|---|---:|---:|---:|---:|
| Pretrain (pre-2019, not scored in objective) | {{HT_PRE_B}} | {{HT_PRE_T}} | {{HT_PRE_D}} | {{HT_PRE_N}} |
| Validation (2019-2022, Optuna saw this) | {{HT_VAL_B}} | {{HT_VAL_T}} | {{HT_VAL_D}} | {{HT_VAL_N}} |
| **Holdout (2022-2024, untouched)** | **{{HT_HOLD_B}}** | **{{HT_HOLD_T}}** | **{{HT_HOLD_D}}** | {{HT_HOLD_N}} |

**Verdict:** {{HT_VERDICT}} (REAL / OVERFITTING / NEUTRAL / NULL)

### 6.2 Per-site holdout Δ R² (the honest number)

| Site | Baseline R² (holdout) | Tuned R² (holdout) | Δ R² | Promote? |
|---|---:|---:|---:|:---:|
| {{HSite1}} | {{HB1}} | {{HT1}} | {{HD1}} | {{HP1}} |
| ... | ... | ... | ... | ... |

### 6.3 Best hyperparameters (only for sites where verdict was REAL)

(Paste from `proposed_overrides.json`. For paper, summarise as "the tuned configurations shifted toward [pattern] consistent with the increased feature count from OAD integration.")

---

## 7. Task 13 — Spike classifier tuning *(if run)*

Objective: F2 (recall-weighted) on `DA > 20 µg/g` events.

| Window | Baseline F2 | Tuned F2 | Δ F2 | Spikes |
|---|---:|---:|---:|---:|
| Validation (2019-2022) | {{SP_VAL_B}} | {{SP_VAL_T}} | {{SP_VAL_D}} | {{SP_VAL_N}} |
| **Holdout (2022-2024)** | **{{SP_HOLD_B}}** | **{{SP_HOLD_T}}** | **{{SP_HOLD_D}}** | {{SP_HOLD_N}} |

| Operating point | Threshold | Holdout precision | Holdout recall |
|---|---:|---:|---:|
| Current | 0.10 | {{SP_CUR_PREC}} | {{SP_CUR_REC}} |
| Tuned | {{SP_NEW_THRESH}} | {{SP_NEW_PREC}} | {{SP_NEW_REC}} |

---

## 8. Limitations & caveats (for the discussion section)

1. **Cloud confound (mild):** OAD scores in SW Washington correlate r = +0.44 with valid-pixel fraction, meaning ~19% of variance is driven by cloud cover rather than ocean state. We mitigate by including 2 cloud-fraction features alongside the 14 score features so the tree ensemble can learn to discount cloudy weeks; alternative is residual-regression (deferred to v2).

2. **MODIS gap 2009-2011:** the cube has zero coverage during these three years (validated against the source parquet). Forecasts in this window have NaN OAD values that the median imputer fills; effectively the model has no OAD signal there. This dilutes any uplift on pooled metrics but does not bias the holdout (2022-2024) where coverage is ~100%.

3. **Lead-1 R² is not the headline:** OAD's RESULTS.md reports R² = +0.87 in SW WA at 1-day lead, but this is inflated by 8-day-composite autocorrelation. At the integration's enforced 12-day lead, the genuine R² is ~0.15. Therefore the expected DA-forecast lift is modest (single-digit R² gains in SW WA, not "doubling baseline performance").

4. **Per-site small-N exclusions:** Coos Bay (N=67), Cannon Beach (N=61), Gold Beach (N=144), Newport (N=142) had OAD intentionally excluded from their handcrafted minimal feature subsets in v1. Task 11 tests whether including OAD as "synthetic data" helps these sites; results in §5.

5. **Hyperparameter tuning protocol:** original per-site values were hand-tuned on a seed=42 dev sample that included future dates, so the published 0.414 dev R² is biased upward. Task 12's chronological holdout produces the unbiased number for any tuned configuration. We report both for completeness.

---

## 9. Reproducibility

**Branch:** `oad-integration` ([GitHub](https://github.com/ansoncchen/DATect-Forecasting-Domoic-Acid/tree/oad-integration))
**Commits to cite in methods:**
- OAD feature module: `6cfa6f72`
- dataset-creation.py wiring: `456da715`
- ablation slot + per-site registration: `f8dfdbd7`
- small-N synthetic experiment: `e538df82`
- hyperparameter tuning infrastructure: `f3690395`
- spike classifier tuning + clip_max addition: `9f6231f0`
- train/val/holdout protocol fix: `bbbac552`
- three-window split (val 2019-2022, holdout 2022-2024): `f380b77b`

**OAD checkpoint used:** `ae_3d_l32_c4_t4_s42_mae050.pt` (Phase C MAE-style training, 50% mask ratio, seed 42). Score parquet derived via `make_ae3d_reconstructor` + `reconstruct_temporal_frame` on the 22-year cube.

**Hyak runs:**
- Task 10 (A/B ablation): job `35390792` on ckpt partition, n3216, ~5 hr total
- Task 11 (small-N): {{HYAK_T11_JOB}}
- Task 12 (tuning array): {{HYAK_T12_JOB}}
- Task 13 (spike tuning): {{HYAK_T13_JOB}}

---

## 10. LaTeX-ready snippets (after numbers are filled)

### Abstract sentence template

```latex
We augment DATect's per-site domoic-acid forecasting ensemble with a 16-dimensional
representation of regional ocean state derived from an unsupervised 3D masked
autoencoder trained on 22 years of MODIS Aqua imagery. On a 2022-2024 temporal
holdout never used in feature selection or hyperparameter tuning, the augmented
ensemble achieves \rsq = {{HOLDOUT_TUNED_R2}} (vs.\ {{HOLDOUT_BASELINE_R2}} for the
per-site environmental baseline; $\Delta = {{HOLDOUT_DELTA_R2}}$, $N = {{HOLDOUT_N}}$),
with the largest improvements concentrated in the SW Washington / Long Beach
region (mean $\Delta\rsq = {{SW_WA_MEAN_DELTA}}$ across four sites).
```

### Table 1 LaTeX skeleton (pooled metrics + ablations)

```latex
\begin{table}[t]
\centering
\caption{Ablation comparison on the random-anchor retrospective evaluation
  (seed 123, $N = {{ABL_BASE_N}}$). Baseline includes per-site customization
  and OAD features; each ablation removes one component.}
\label{tab:ablations}
\begin{tabular}{lrrr}
\toprule
Configuration & \rsq & $\Delta\rsq$ & MAE \\
\midrule
Baseline (full DATect + OAD)    & {{ABL_BASE_R2}} & ---              & {{ABL_BASE_MAE}} \\
\quad -- No interpolated training & {{ABL_NI_R2}}   & {{ABL_NI_DELTA}} & {{ABL_NI_MAE}}   \\
\quad -- No per-site customization& {{ABL_NP_R2}}   & {{ABL_NP_DELTA}} & {{ABL_NP_MAE}}   \\
\quad -- No observation-order lags& {{ABL_NL_R2}}   & {{ABL_NL_DELTA}} & {{ABL_NL_MAE}}   \\
\quad -- No derived features      & {{ABL_ND_R2}}   & {{ABL_ND_DELTA}} & {{ABL_ND_MAE}}   \\
\quad -- \textbf{No OAD features} & \textbf{{{ABL_NO_R2}}} & \textbf{{{ABL_NO_DELTA}}} & \textbf{{{ABL_NO_MAE}}} \\
\bottomrule
\end{tabular}
\end{table}
```

### Figure ideas

1. **Holdout per-site bar chart** — `baseline R²` and `+OAD R²` side-by-side for all 10 sites, colored by region, highlighting SW WA.
2. **Validation vs holdout scatter** — one dot per Task 12 trial, x = val R², y = holdout R². Slope < 1 shows the overfitting gap; should be near 1 if tuning generalises.
3. **OAD score vs DA time series for one SW WA site** (e.g., Twin Harbors 2014-2024) — overlay the daily OAD `oad_score` with the weekly DA measurements and the model's spike alerts. Demonstrates the lead-lag relationship visually.

---

## 11. Post-Task-10 diagnostic: OAD ↔ DA correlation

A null A/B result for OAD (Δ R² = −0.0015 pooled) prompted a direct correlation check between the raw `oad_score` and DA measurements. All per-site |Pearson r| < 0.12; pooled r = −0.046. Among spike events (DA > 20 µg/g), mean OAD is *lower* than during non-spikes (2.91 vs 3.27).

**Lagged correlation reveals signal at ~12 weeks:**

| Site | r at lag=0w | r at lag=4w | r at lag=12w |
|---|---:|---:|---:|
| Cannon Beach | +0.093 | +0.105 | **+0.183** |
| Kalaloch | +0.042 | +0.066 | **+0.155** |
| Twin Harbors | −0.041 | +0.002 | **+0.144** |
| Long Beach | −0.049 | −0.016 | +0.106 |
| Quinault | +0.026 | +0.017 | +0.089 |
| Copalis | −0.017 | +0.020 | +0.085 |
| Clatsop Beach | −0.008 | −0.019 | +0.084 |
| Coos Bay | −0.124 | −0.130 | −0.068 |
| Newport | −0.026 | −0.046 | −0.046 |
| Gold Beach | −0.111 | −0.100 | −0.111 |

**Interpretation:** the v1 integration uses OAD at 1-3 week lags (R−5 to R−19), capturing essentially none of the 12-week WA-site signal. Oregon sites (Coos Bay, Newport, Gold Beach) stay negatively correlated at all lags, consistent with their established autocorrelation-ceiling R² ≈ 0.

**v2 design hint:** add `oad_score_90day_mean`, `oad_score_180day_max` with the same R−5 leak shift to capture the longer-timescale upwelling-priming signal. Not changing v1 — this experiment was a clean test of the short-lag formulation.

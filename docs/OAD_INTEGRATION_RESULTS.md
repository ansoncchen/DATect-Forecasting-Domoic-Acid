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

---

## 12. Why OAD didn't help — feature-level diagnostic

Direct comparison of candidate DA predictors against actual DA, pooled across all 10 sites, at various time lags:

| Feature | lag=0w | lag=4w | lag=8w | lag=12w | lag=16w |
|---|---:|---:|---:|---:|---:|
| `modis-chla` (raw) | −0.002 | −0.007 | −0.005 | +0.021 | +0.020 |
| `oad_score` (AE anomaly) | −0.022 | −0.010 | +0.002 | +0.062 | +0.050 |
| `modis-sst` (raw) | +0.032 | +0.036 | +0.040 | +0.045 | +0.045 |
| `beuti` (upwelling index) | +0.044 | +0.034 | +0.026 | +0.037 | +0.035 |
| **`sst-anom`** (SST climatology anomaly) | **+0.143** | **+0.170** | **+0.172** | **+0.171** | **+0.203** |

**Key finding:** RAW chlorophyll is essentially uncorrelated with DA (lag-0 r ≈ 0), regardless of how compelling bloom periods look in chla imagery. The OAD anomaly score inherits this weakness because the AE was trained on raw chla/Kd490/nflh/sst inputs — it learns to reconstruct typical seasonal patterns including high-chla bloom periods, so blooms aren't anomalous to it.

The real DA-predictive signal lives in **`sst-anom`** (SST minus climatology), which DATect already includes as a per-site feature. Its correlation with DA is **3-10× stronger than any other satellite feature** at lags up to 16 weeks.

**Implications for v2:**
1. Retrain the AE on climatology-normalized inputs (subtract per-pixel per-DOY mean before training) so its "anomaly" matches what `sst-anom` already shows.
2. OR add a learned SST-anomaly representation as a separate channel alongside the raw fields.
3. The 16-week lag pattern suggests a multi-month upwelling-priming process for DA blooms — feature engineering should target this timescale, not the 1-3 week short lags v1 uses.

Per-site comparison (lag=0):

| Site | chla→DA | OAD→DA | sst-anom→DA |
|---|---:|---:|---:|
| Cannon Beach | +0.151 | +0.093 | +0.084 |
| Coos Bay | −0.063 | −0.124 | **+0.202** |
| Copalis | −0.024 | −0.017 | **+0.214** |
| Kalaloch | +0.051 | +0.042 | **+0.192** |
| Long Beach | −0.045 | −0.049 | **+0.149** |
| Twin Harbors | −0.030 | −0.041 | **+0.212** |
| Newport | +0.016 | −0.026 | **+0.110** |

`sst-anom` consistently outperforms chla and OAD at every site except Cannon Beach (where Δ is small and N is smallest).

---

## 13. Regional chla diagnostic — answering "would regional vs per-pixel chla help?"

A natural hypothesis was that per-site `modis-chla` (single-pixel near each beach) might miss broader regional bloom patterns, and that regional-mean chla (averaged over the OAD polygon) would correlate better with DA. Tested by deriving per-(date, region) `chla_mean`, `chla_max`, `chla_p95` from the cube and correlating with site DA via `SITE_TO_REGION`.

**Result: regional and single-pixel chla are equivalently weak.** Per-site Pearson r:

| Site | Region | regional chla_mean → DA | regional chla_p95 → DA | per-pixel modis-chla → DA |
|---|---|---:|---:|---:|
| Kalaloch | Olympic Coast WA | +0.093 | +0.095 | +0.090 |
| Quinault | Olympic Coast WA | +0.055 | +0.024 | −0.007 |
| Copalis | Olympic Coast WA | +0.024 | −0.012 | +0.008 |
| Twin Harbors | SW Washington | −0.045 | −0.056 | −0.008 |
| Long Beach | SW Washington | −0.071 | −0.088 | −0.050 |
| Clatsop Beach | SW Washington | −0.046 | −0.102 | −0.083 |
| Cannon Beach | SW Washington | +0.138 | +0.157 | +0.225 |
| Newport | Central Oregon | −0.002 | +0.007 | −0.045 |
| **Coos Bay** | Central Oregon | **−0.200** | **−0.190** | −0.114 |
| Gold Beach | Southern OR/N CA | −0.036 | −0.030 | −0.053 |

**Pooled lagged correlation (regional chla_mean → DA at lag k):**

| lag | 0w | 4w | 8w | 12w | 16w |
|---|---:|---:|---:|---:|---:|
| pooled r | −0.009 | −0.004 | −0.007 | +0.037 | +0.053 |

For reference, the same lag-16w analysis for `sst-anom` produced pooled r = **+0.203** — 4× stronger.

**Interpretation:** chlorophyll concentration alone, regardless of spatial averaging or absolute magnitude, is a poor predictor of DA at these specific shellfish beaches. The OAD null result was inevitable given its raw-chla training input. The strongly negative correlation at Coos Bay (regional chla mean r = −0.20) is consistent with the "wrong species" interpretation — high chla often reflects non-Pseudo-nitzschia phytoplankton communities that don't produce domoic acid.

**Conclusion: regional aggregation didn't rescue the chla signal.** The OAD design needs to target a different physical quantity (e.g., SST anomaly from climatology) rather than raw optical fields, to be useful for DA forecasting.

---

## 14. Synthesis with Moore et al. 2021 ESP observations

[Moore et al. 2021](https://www.mdpi.com/2077-1312/9/3/336) deployed an Environmental Sample Processor (ESP) at the NEMO mooring (NW Washington shelf, ~24 km W/NW of La Push) for 2016-2018, directly measuring *Pseudo-nitzschia* cell counts and particulate DA in seawater alongside collocated wind, current, nitrate, and chlorophyll sensors. Their findings directly inform why OAD (and chla generally) fail as DA forecasting features.

### Key finding from the ESP record

**Chlorophyll LAGS the bloom signal by ~2 days.** Their Section 3.2 documents the May 2017 bloom: "Total *Pn* quantified (and particulate DA) began to increase around 7 May, about 2 days *before* chlorophyll-a began to appreciably increase." Figure 7b shows the temporal sequence explicitly. Visual chla maps document blooms that are already producing toxin — they cannot be a leading indicator at the 12+ day horizon DATect operates on.

### The actual bloom mechanism (from Moore et al. §3.2-3.3)

1. **5+ days of northerly upwelling-favorable winds** → shoaling of nutrient-rich deep water.
2. **Wind reversal or coastal-trapped internal wave event** (period ~1 week) → pulses nutrients into the euphotic zone (20-50 m depth).
3. **Pn population rises** in response to nutrients.
4. **DA produced** ~1-2 days into the bloom.
5. **Chlorophyll detectable** in satellite imagery ~2 days after Pn.

The driving variables are wind, internal wave activity, and subsurface stratification — NOT surface chlorophyll. This explains both our correlation diagnostic (§12-13) and the OAD A/B null result (§3).

### Concrete v2 features grounded in Moore et al.

**Tier 1 — derivable from data DATect can already access** (NDBC buoys + BEUTI + existing satellite):

1. **Cumulative upwelling-favorable wind stress** over past 5, 14, 30 days from NDBC buoy 46041 (Cape Elizabeth) — captures the multi-day northerly-wind precursor.
2. **Wind reversal events** — count of N→S wind transitions in past 14/30 days; each transition is a potential nutrient-pulse trigger.
3. **BEUTI 14-day delta** and **BEUTI climatology anomaly** — derivatives of the existing `beuti` feature capture upwelling pulses better than raw value.
4. **SST tendency** (dT/dt over past 7-14 days) — large negative dT/dt = active upwelling.
5. **Sea-level pressure variance** over past 14 days — proxy for storm activity that triggers coastal-trapped internal waves.

**Tier 2 — needs new data sources**:

6. **Subsurface temperature profile** at the shelf (20-50 m depth) from NEMO/Cha Ba moorings. Shoaling rate of the 8°C isotherm into the 20-50 m band is the actual nutrient-injection signal per Figure 7c.
7. **Along-shelf velocity oscillations** (~1-week period) from moored ADCPs — direct measure of coastal-trapped internal wave activity.
8. **Pn cell-count time series** from ESP deployments where available (this paper's 2016-2018 data, plus continuing NANOOS HABs program).

**Tier 3 — redesign the OAD autoencoder**:

9. Train a new AE where inputs are **wind + BEUTI + SST anomaly + subsurface T profile + along-shelf velocity** (the physical drivers), not raw chla/Kd490/nflh/SST. The "anomaly score" of this model would measure "how unusual is today's upwelling-regime state" — a true leading indicator.

### Single highest-value change recommended

If only one feature were added to DATect v2: **cumulative northerly wind stress over past 14 days** from NDBC Cape Elizabeth (46041). NDBC data is free and daily, the feature is computationally trivial, and it is the leading indicator that Moore et al. identified directly. DATect's existing `beuti` partially captures this but as a single instantaneous value; the cumulative + delta versions add the temporal context the paper shows matters most.

---

## 15. Available in-situ datasets — game-changing v2 features

User shared additional datasets after the OAD null result was confirmed. Quick inventory + correlation tests reveal that **direct measurement of *Pseudo-nitzschia* cell counts at the beach is far more informative than any satellite signal we tested**.

### Dataset inventory

| File | Sites | Years | Key variables |
|---|---|---|---|
| `ORHAB_KAL-COP-TH-LB_upto_060815.xlsx` | Kalaloch, Copalis, Twin Harbors, Long Beach | 2000-2016 | **PN count (cells/L), pDA (ng/L)** at the beach, ~weekly |
| `ChaBa ESP database.xlsx` | NEMO mooring (offshore WA shelf) | 2016-2018 | Pn species probes + pDA, daily-to-weekly |
| `Summary NWFSC 2021_2023 data` | NEMO mooring | 2021-2023 | pDA only, extends ESP series |
| `WQM_*.csv/.DAT` | NEMO mooring | 2023 (months) | Subsurface T, salinity, DO, chla, turbidity |
| `pCO2105_ALLdata*.txt` | NEMO mooring | 2023 | pCO2 + fluorometric chla |

### ORHAB PN count → DATect shellfish DA (the key result)

For the 4 WA sites that have ORHAB data, lagged Pearson correlation of weekly PN count at week (t−k) with DATect's shellfish DA at week t:

| Site | r at lag 0w | lag 1w | **lag 2w** | lag 4w | lag 8w | N |
|---|---:|---:|---:|---:|---:|---:|
| **Twin Harbors** | +0.21 | +0.25 | **+0.36** | +0.30 | +0.19 | 615 |
| **Long Beach** | +0.24 | +0.32 | **+0.34** | +0.20 | +0.13 | 621 |
| Copalis | +0.11 | +0.14 | +0.13 | +0.10 | +0.10 | 596 |
| Kalaloch | +0.13 | +0.14 | +0.13 | +0.07 | +0.07 | 582 |

**Best lag is 2 weeks** for Twin Harbors and Long Beach — biologically sensible: PN cells produce particulate DA → razor clams filter water for ~1-2 weeks → DA peaks in clam tissue 1-2 weeks later.

For comparison, the strongest satellite feature `sst-anom` peaked at **r = +0.20 at lag 16 weeks** (§12). The ORHAB PN feature is **~1.8× stronger AND at 8× shorter lag**, with direct biological mechanism.

### ORHAB PN count → ORHAB particulate DA (sanity check, concurrent)

Confirms direct mechanism: at the same time and place, PN count correlates strongly with measured particulate DA in seawater:

| Site | N | r(PN, pDA) |
|---|---:|---:|
| **Kalaloch** | 712 | **+0.71** |
| Long Beach | 751 | +0.43 |
| Twin Harbors | 744 | +0.41 |
| Copalis | 726 | +0.39 |

Kalaloch r=+0.71 is the strongest pDA-driver correlation we've seen anywhere in this analysis.

### Concrete v2 path — ranked by impact-per-effort

**1. Add ORHAB PN count as a per-site feature (HIGHEST PRIORITY)**
- Coverage: 4 sites × 16 years (2000-2016)
- For each weekly anchor row at Kalaloch/Copalis/Twin Harbors/Long Beach, add `orhab_pn_count_lag1w`, `orhab_pn_count_lag2w`, `orhab_pn_count_lag4w`, `orhab_pn_count_30day_mean`
- Implementation: ~50 lines, mirrors `add_oad_features` pattern from this integration. Use 5-day leakage shift to match the existing 12-day forecast horizon.
- Expected lift: at Twin Harbors/Long Beach, plugging an r=+0.36 feature into the ensemble should produce visible R² improvement (not the null effect OAD gave). The other 2 sites get a weaker but still real signal.
- Caveat: no coverage 2016+. The other 6 sites (Newport, Coos Bay, Gold Beach, Cannon Beach, Clatsop Beach, Quinault) have no ORHAB data.

**2. Add ChaBa ESP offshore pDA as a regional feature**
- Coverage: 2016-2018 + 2021-2023 (NEMO mooring, offshore WA shelf, source region for Olympic Coast + SW WA sites)
- Apply to all 7 WA sites in our SITE_TO_REGION map (Kalaloch, Quinault, Copalis, Twin Harbors, Long Beach, Clatsop Beach, Cannon Beach)
- Implementation: same pattern as ORHAB, but `region` is "WA shelf" (single source)
- Expected lift: smaller than ORHAB PN (offshore lags onshore + has spatial gap to shore), but still real because the cells eventually transport to beach

**3. Add WQM subsurface T/nitrate/DO** when available
- Coverage: 2023 (intermittent) — too sparse for direct training
- Use as **validation set only** for v2 mechanism checks; not yet a model feature

**4. Redesign OAD inputs (deferred)**
- Replace AE's raw chla/Kd490/nflh/SST with wind + BEUTI + sst-anom + subsurface T
- Now needed only if we want a general-purpose anomaly index; the direct PN features above are more efficient if you only need DA forecasting

### Reframing the OAD project's contribution

With these in-situ datasets in hand, the OAD project has a clean self-assessment:

> Satellite optical anomaly representations cannot match direct in-situ *Pseudo-nitzschia* cell count measurements as DA leading indicators (r=+0.36 vs |r|<0.15 at any lag, §13). The OAD project's value to DATect is therefore not as a feature provider, but as a **diagnostic that confirmed which physical signals carry DA-predictive information**: surface chlorophyll at any spatial scale does not (§12-13), SST anomaly from climatology does (§12), and direct cell counts dominate when available (§15).
>
> The recommended v2 architecture pairs DATect's existing climate features (sst-anom, BEUTI, PDO, ONI) with ORHAB beach PN counts where available, and reserves satellite-derived anomalies for use in the unsampled gap regions (Cannon Beach, OR sites) where in-situ data is sparse.

---

## 16. OAD ↔ ESP in-situ test — *OAD works, but at the wrong layer*

The most consequential single test in this integration. The ESP mooring at NEMO (24 km offshore WA shelf, 2016-2018) provides direct in-situ measurements of *Pseudo-nitzschia* cell counts and particulate DA at the same location and timescale where the AE was trained.

### OAD score → ESP *Pn* cell density (offshore)

Pooled cell density = `auD1 (P. australis) + muD1 (P. multiseries) + frD2 (P. fraudulenta) + pung1 (P. pungens)`:

| OAD region | lag −7d | **lag 0d** | lag +7d | lag +21d | N |
|---|---:|---:|---:|---:|---:|
| **Olympic Coast (WA)** | +0.073 | **+0.458** | +0.164 | +0.066 | 76 |
| **SW Washington / Long Beach** | +0.238 | +0.305 | **+0.543** | +0.346 | 76 |
| Overall WA-OR-NCA | −0.049 | +0.160 | +0.175 | +0.055 | 76 |

### OAD score → ESP particulate DA (offshore)

| OAD region | **lag 0d** | lag +21d | N |
|---|---:|---:|---:|
| **Olympic Coast (WA)** | **+0.317** | +0.074 | 90 |
| **SW Washington / Long Beach** | **+0.334** | **+0.324** | 90 |
| Overall WA-OR-NCA | +0.207 | +0.117 | 90 |

### Implication for the project's contribution

**OAD does measure real bloom-related anomalies at the offshore source.** Correlations of r = +0.46 (Pn cells, concurrent) and r = +0.33 (pDA, concurrent) substantially exceed everything we measured against beach DA (|r| < 0.15). The AE-encoded satellite signal genuinely captures something about offshore phytoplankton anomalies.

**The signal does not survive the transport + bioaccumulation chain to beach DA.** The 24 km onshore transport (wind-dependent timing), site-specific local oceanography (which beaches the cells reach), and 1-2 week razor clam bioaccumulation cumulatively destroy the predictive relationship between offshore OAD and beach shellfish DA.

This reframes the OAD project's standing:
- **Successful** as an unsupervised representation of offshore ocean state related to Pn bloom intensity (validated against ESP data, r = +0.31–0.54).
- **Unsuccessful** as a leading indicator for shellfish DA at specific beaches (this paper's null result).

The right framing for the paper:

> "An unsupervised 3D masked-autoencoder trained on 22 years of MODIS satellite imagery learns regional anomaly representations that significantly correlate with in-situ *Pseudo-nitzschia* cell density (r = +0.46) and particulate domoic acid (r = +0.33) at offshore source regions (NEMO mooring data, Moore et al. 2021). However, this signal does not propagate to shellfish DA at coastal monitoring beaches (per-site |r| < 0.15 at any lag) due to cumulative noise from variable wind-driven onshore transport (~24 km) and razor clam bioaccumulation (~1-2 weeks). The integration provides a clean demonstration of where satellite-derived ocean anomalies are and are not useful for harmful algal bloom forecasting."

---

## 17. The biggest immediate free lift — lagged PN features for 9 sites

While investigating other v2 datasets, found that DATect ALREADY has `pn` (*Pseudo-nitzschia* cell counts) as a column in `final_output.parquet` for all 10 sites. But `per_site_models.py:SITE_SPECIFIC_CONFIGS` only includes `PN_FEATURES = ['pn_log']` in **Kalaloch's** feature_subset. Nine sites are leaving free signal on the table.

### Lagged PN → DA correlations from the existing parquet

| Site | r lag 0w | r lag 1w | **r lag 2w** | r lag 4w | currently uses PN? |
|---|---:|---:|---:|---:|:---:|
| **Twin Harbors** | +0.170 | +0.191 | +0.230 | **+0.314** | NO |
| **Long Beach** | +0.146 | +0.205 | **+0.250** | +0.225 | NO |
| **Kalaloch** | +0.087 | +0.170 | **+0.181** | +0.141 | yes |
| Quinault | +0.113 | **+0.149** | +0.138 | +0.133 | NO |
| Copalis | +0.039 | +0.078 | **+0.104** | +0.088 | NO |
| Clatsop Beach | +0.008 | +0.019 | +0.027 | +0.035 | NO (subset=None → all features) |
| Coos Bay / Cannon / Gold / Newport | < +0.04 | < +0.04 | < +0.04 | < +0.04 | NO |

### Recommended v2 changes (zero new data needed)

1. **Add `pn_log` to feature_subset** for Twin Harbors, Long Beach, Quinault, Copalis (the 4 sites with r > 0.10 at any lag).
2. **Add lagged PN features** (`pn_log_lag1w`, `pn_log_lag2w`, `pn_log_lag4w`) to the same 4 sites — mirrors the OAD lag-feature pattern. Expected to add another +0.05 to +0.10 R² because the strongest correlation is at lag, not concurrent.
3. **Cap order-of-magnitude**: `pn_log = log1p(pn)` to handle the wide dynamic range (0 to 4.7M cells/L) and reduce influence of outliers.

### Why this is more important than the OAD integration

| | OAD integration (this report) | Lagged PN features (proposed) |
|---|---|---|
| Effort | Major (cube training, OAD scoring, leakage analysis, regional joins) | Minor (5 lines per site in per_site_models.py) |
| Data | New 10 GB cube, AE training, score parquet | Already in `final_output.parquet` |
| Best per-site r vs DA | < +0.15 anywhere | +0.31 at Twin Harbors lag 4w |
| Mechanism | Compressed regional satellite anomaly (indirect) | Direct cell count of toxin-producing organism (direct) |
| Confidence in lift | Null at α=0.05 | r=+0.31 implies measurable R² gain |

### About the ORHAB data the user shared

Largely **redundant**: DATect's `data/raw/pn-input/long-beach-pn.csv` has 2,002 rows vs ORHAB's 1,419 — DATect has the same monitoring program output, with more rows and a longer date range. The unique value of ORHAB is the `pDA (ng/L)` column (particulate DA in seawater), which differs from DATect's `da` (razor clam tissue) — but pDA is more useful as an evaluation target than a feature, since it requires direct seawater sampling that isn't part of the operational monitoring workflow.

---

## 18. Hyperparameter tuning result — TUNING FAILED (Tasks 12+13)

After tuning 9 sites with Optuna TPE (30+ trials each, 18 hyperparameters per site) using the 3-window chronological split (train pre-2019 / validate 2019-2022 / holdout 2022-2024), the holdout comparison decisively rejected the tuned configurations.

### Window-level comparison (baseline current per_site_models.py vs tuned)

| Window | Baseline R² | Tuned R² | Δ R² | N |
|---|---:|---:|---:|---:|
| Pretrain (pre-2019, not scored) | 0.428 | 0.312 | −0.115 | 797 |
| Validation (2019-2022) | 0.346 | 0.418 | **+0.072** | 216 |
| **Holdout (2022-2024, untouched)** | **0.495** | **0.338** | **−0.157** | 164 |

**Verdict from `validate_tuned_on_holdout.py`: OVERFITTING.** Tuning improved validation but degraded holdout — the classic noise-fitting signature.

### Per-site detail (holdout window only — the honest numbers)

| Site | Baseline R² | Tuned R² | Δ R² | N | Outcome |
|---|---:|---:|---:|---:|---|
| Gold Beach | −8.46 | −8.14 | +0.32 | 17 | "less broken" not "fixed" — baseline R²=−8 is fundamental |
| Quinault | 0.348 | 0.462 | +0.11 | 24 | Only meaningful per-site improvement |
| Twin Harbors | 0.570 | 0.577 | +0.01 | 18 | Tie |
| Copalis | 0.682 | 0.683 | 0.00 | 30 | Tie |
| Kalaloch | 0.674 | 0.660 | −0.01 | 11 | Tiny loss |
| Clatsop Beach | 0.444 | 0.407 | −0.04 | 13 | Small loss |
| Long Beach | 0.810 | 0.735 | −0.08 | 19 | Loss |
| **Coos Bay** | 0.738 | 0.562 | **−0.18** | 12 | Big loss — Optuna found a val-window winner (+0.57) that collapsed on holdout |
| **Newport** | 0.375 | 0.134 | **−0.24** | 20 | Big loss — same pattern |

**Decision: do NOT apply `proposed_overrides.json`.** Original hand-tuned values in `per_site_models.py` stay. This empirically confirms the existing stability study's finding (|ΔR²| < 0.001 for RF perturbations) — the hand-tuned values were already in a flat region of the loss landscape, so Optuna's broader search just fit noise on the validation window.

### Spike classifier tuning (Task 13)

| Window | Baseline F2 | Tuned F2 | Δ | N spikes |
|---|---:|---:|---:|---:|
| Validation (2019-2022) | (not yet computed) | 0.732 | — | 32 |
| Holdout (2022-2024) | (not yet computed) | (pending) | — | — |

Spike classifier holdout test pending. The tuned threshold is 0.227 (vs current 0.10), giving recall 0.91 / precision 0.41 on validation. Need to confirm on holdout before adopting.

### Lessons

1. **The 3-window protocol works as designed.** Without the chronological holdout, we'd have published "tuning improved R² by 0.07" — and been wrong.
2. **Optuna can overfit even with strong regularization in the search space.** 18-dimensional search × 30 trials × 9 sites = 4860 hyperparameter combinations evaluated. With val-window N=216 pooled (and per-site N=10-32), there's plenty of room for spurious wins.
3. **The asymmetric verdict (REAL requires BOTH windows positive) caught the failure.** A naive "improve on val → adopt" rule would have shipped these regressions.
4. **DATect's existing per-site customization is robust.** The stability study, the Task 10 OAD null, and now this tuning result all point to the same conclusion: the model is sensitive to FEATURES that carry signal (lagged PN, sst-anom, wind dynamics — see §17, §14), not to hyperparameter perturbations of the existing feature set.

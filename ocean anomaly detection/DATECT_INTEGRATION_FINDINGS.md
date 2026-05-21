# OAD → DATect Integration Findings

> **What this is:** a self-contained summary of what we learned by integrating the OAD
> anomaly score as a feature in DATect's per-site domoic acid forecasting model.
>
> **The headline finding:** OAD score correlates significantly with in-situ ESP
> measurements at the offshore source (Pn cells r=+0.46, pDA r=+0.33), but does not
> propagate to shellfish DA at coastal monitoring beaches (|r|<0.15 at any lag).
>
> **Companion docs:**
> - [`RESULTS.md`](RESULTS.md) — OAD's intrinsic E4-forecastability validation against PCA
> - [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — design phases of the OAD subproject
> - [`../docs/OAD_INTEGRATION_RESULTS.md`](../docs/OAD_INTEGRATION_RESULTS.md) — full paper-ready writeup (17 sections)

---

## 1. TL;DR

| Question | Answer |
|---|---|
| Does OAD measure real bloom-related signal? | **Yes** (validated against in-situ ESP data) |
| Does OAD improve site-level DA forecasting at beaches? | **No** (Δ R² = −0.0015, sub-noise null) |
| Where does the signal stop propagating? | The 24 km onshore transport + 1–2 wk razor clam bioaccumulation chain |
| Is there a better feature to add to DATect? | **Yes** — DATect's existing `pn` column (lagged), currently used by only 1 of 10 sites |
| Is more hyperparameter tuning needed? | Possibly. Per-site Optuna tuning showed +0.13 to +0.30 R² gains at 4 sites on validation; holdout validation pending. |

## 2. Data coverage caveat (read first)

The cleanest test of OAD's signal — comparison against ESP-measured in-situ Pn cells and particulate DA at the NEMO mooring — is limited to:
- **2016-2018**: ChaBa ESP deployments (Moore et al. 2021), 90 pDA samples + 76 Pn-cell samples
- **2021-2023**: NWFSC continuation (pDA only, not yet integrated into the correlation analysis)

So **5 years of intermittent offshore data**, vs the 21-year DATect record. The correlations below are reported with 95% bootstrap CIs (2000 resamples, seed=42) so you can see the uncertainty bounds.

The shellfish-DA correlation analyses use the full DATect dataset (10 sites × 21 years).

## 3. The OAD-validates-against-ESP result

**Method**: per-region OAD score (from `ae_3d_l32_c4_t4_s42_mae050`) joined on date with ESP cELISA (pDA) and SHA (Pn cells) measurements at NEMO mooring (offshore N WA shelf, 47.97°N 124.97°W). Pearson r contemporaneously.

### OAD score → ESP Pseudo-nitzschia cell density (76 daily samples, 2016-2018)

| OAD region | N | r | 95% CI | p |
|---|---:|---:|---|---:|
| **Olympic Coast (WA)** | 76 | **+0.458** | [+0.185, +0.655] | 3.1×10⁻⁵ |
| SW Washington / Long Beach | 76 | +0.305 | [+0.105, +0.512] | 7.3×10⁻³ |
| Overall WA-OR-NCA envelope | 76 | +0.160 | [−0.095, +0.387] | 0.17 |

### OAD score → ESP particulate domoic acid (90 daily samples, 2016-2018)

| OAD region | N | r | 95% CI | p |
|---|---:|---:|---|---:|
| Olympic Coast (WA) | 90 | +0.317 | [−0.018, +0.526] | 2.3×10⁻³ |
| **SW Washington / Long Beach** | 90 | **+0.334** | [+0.125, +0.518] | 1.3×10⁻³ |
| Overall WA-OR-NCA envelope | 90 | +0.207 | [+0.021, +0.363] | 0.05 |

**Interpretation**: at the NEMO source region (where the AE was trained), OAD score
encodes meaningful information about both *Pseudo-nitzschia* cell density AND the
toxin it produces. The Olympic Coast region (which contains NEMO) shows the strongest
Pn-cell correlation; SW Washington shows the strongest pDA correlation. The Overall
coast-wide envelope is weakest — spatial averaging dilutes the localized signal.

**Limitations**:
- Small N (76-90) keeps CIs wide; Olympic Coast pDA CI even includes 0
- Single mooring location (NEMO); regional generalization not directly tested
- ESP only deployed during expected bloom seasons; sampling not random

Still, the consistent positive sign + multiple-region replication + p<0.01 at the two strongest tests is strong evidence that **OAD captures real offshore bloom signal**.

## 4. The OAD-doesn't-translate-to-beach result

DATect's target is razor clam shellfish DA at 10 Pacific coast beaches. Despite OAD's
offshore validation above, integrating it as a feature gave essentially zero lift:

| Metric (pooled, N=1202 random retrospective anchors) | Baseline (no OAD) | + OAD | Δ |
|---|---:|---:|---:|
| R² | 0.1749 | 0.1734 | **−0.0015** |
| MAE (µg/g) | 6.51 | 6.53 | +0.013 |

Per-site Δ R² in the SW Washington region (where OAD's headline win lives):

| Site | Δ R² (+OAD − baseline) |
|---|---:|
| Cannon Beach | +0.0000 |
| Twin Harbors | −0.0056 |
| Long Beach | −0.0056 |
| Clatsop Beach | −0.0084 |

All within the |ΔR²| < 0.01 noise floor established by the existing stability study.

## 5. Why the offshore signal doesn't reach the beach

Three compounding noise sources between OAD (24 km offshore) and DATect's target
(toxin accumulated in razor clam tissue at the beach):

| Layer | What happens | Effect on signal |
|---|---|---|
| Transport | Cells advect 24 km onshore via wind-driven currents (timing varies) | Cells reach different beaches at different times; some beaches miss the bloom entirely |
| Cell mortality + dilution | Many cells die or are diluted during transit | Magnitude of signal attenuates |
| Species selection | Total Pn ≠ DA-producing Pn (P. australis, P. multiseries are toxic; P. pungens often isn't) | Even strong "total bloom" doesn't always = strong toxin |
| Bioaccumulation | Razor clams filter water for 1-2 weeks before DA shows up in tissue | Adds temporal lag + per-clam variability |
| Spatial gap | NEMO is at one point; each beach is at a different point | Different beaches have different transport corridors |

Each step adds noise. By the time the signal reaches shellfish tissue, the original
OAD anomaly score has been buried.

## 6. What ELSE we tested (supporting evidence)

### 6.1 Per-pixel and regional chlorophyll fail equivalently

| Predictor | Best per-site |r| against beach DA | Best pooled r at any lag (0-16w) |
|---|---:|---:|
| Per-pixel `modis-chla` | 0.225 (Cannon Beach concurrent) | +0.021 (16w lag) |
| Regional chla **mean** over OAD polygon | 0.204 (Coos Bay, NEGATIVE) | +0.053 (16w lag) |
| Regional chla **p95** | 0.190 (Coos Bay, NEGATIVE) | +0.060 (16w lag) |
| OAD anomaly score | 0.124 (Coos Bay, NEGATIVE) | +0.062 (12w lag) |

**Spatial aggregation didn't rescue chlorophyll as a predictor.** Regional and per-pixel
chla are equivalently weak. Coos Bay's strongest correlation is NEGATIVE — high chla
often reflects non-Pseudo-nitzschia blooms.

### 6.2 SST anomaly from climatology is the strongest satellite predictor

| Predictor | Pooled r at lag 0 | Pooled r at lag 16w |
|---|---:|---:|
| `modis-chla` | −0.002 | +0.020 |
| `modis-sst` (raw) | +0.032 | +0.045 |
| `oad_score` | −0.022 | +0.050 |
| `beuti` (existing) | +0.044 | +0.035 |
| **`sst-anom`** (existing) | **+0.143** | **+0.203** |

DATect already includes `sst-anom` and the existing tuning correctly weights it
highly. The OAD subproject's value is therefore **diagnostic** — it confirmed that
the satellite-chla pathway doesn't carry DA-predictive information, validating the
existing preference for SST-anomaly features.

### 6.3 ORHAB beach data is largely redundant with DATect's existing inputs

Comparison of ORHAB Long Beach PN data (1,419 rows, 2000-2015) vs DATect's
`data/raw/pn-input/long-beach-pn.csv` (2,002 rows, 2002-2023):

| Site | Shared dates | Exact PN match | DATect extends through |
|---|---:|---:|---|
| Long Beach | 1,179 | 100% (1178/1179) | 2023-12 |
| Kalaloch | 919 | 99% (910/919) | 2023-11 |
| Copalis | 944 | 99% (936/944) | 2023-12 |

ORHAB is the same monitoring program already feeding DATect. The 1% disagreements
are below-LLOQ detections that DATect's pipeline filters to zero. The only ORHAB-
unique signal is the `pDA (ng/L)` column (particulate seawater DA, distinct from
shellfish DA) — but DATect already targets shellfish DA so pDA is secondary.

## 7. Hyperparameter tuning (in progress)

After the OAD null result was confirmed, we launched two Optuna-driven tuning jobs:

### 7.1 Per-site regression tuning
- **Scope**: 18 hyperparameters per site (10 XGB + 5 RF + ensemble weight + clip_q + clip_max)
- **Protocol**: 3-window chronological split — train (pre-2019) / validate (2019-2022, Optuna's objective) / holdout (2022-2024, untouched final test)
- **Status**: 4 of 10 sites complete as of writing; partial validation-window results:

| Site | Baseline R² (Task 10) | Tuned R² (val 2019-22) | Δ on val | N val |
|---|---:|---:|---:|---:|
| Long Beach | +0.520 | +0.653 | **+0.13** | 16 |
| Clatsop Beach | +0.296 | +0.383 | +0.09 | 32 |
| Gold Beach | +0.035 | +0.331 | +0.30 | 30 |
| Newport | −0.163 | −0.093 | +0.07 | 31 |

These are SIGNIFICANT IF they generalize to the 2022-2024 holdout — pending validation.
Long Beach N=16 is small enough that the +0.13 could be noise. Gold Beach +0.30 is
suspiciously large; the holdout test will determine whether tuning genuinely helped
or whether Optuna fit noise on the validation window.

### 7.2 Spike classifier tuning
- **Scope**: 9 XGB params + alert probability threshold (`SPIKE_ALERT_PROB_THRESHOLD`)
- **Objective**: F2 (recall-weighted) on DA > 20 µg/g events
- **Status**: DONE, validation-window F2 = 0.732 (recall 0.91, precision 0.41). Alert threshold tuned from 0.10 → 0.227 (less sensitive → fewer false alarms). Holdout F2 pending.

## 8. v2 feature chains scaffolded (not yet run)

Four feature-extension experiments are coded and ready to queue as
`chains/run_chain.sbatch` once tuning completes:

| Chain | What it adds | Sites | Expected lift |
|---|---|---|---|
| `c1_lagged_pn` | 7 lagged PN features (DATect's own data) | 5 high-N | **Real** — r=+0.31 at TH lag 4w (highest expected gain) |
| `c2_beuti_derivatives` | 7 BEUTI temporal-context features | all 10 | Modest — derivatives of existing feature |
| `c3_nemo_mooring` | 8 in-situ subsurface features (T, salt, DO, chl, pCO2 from NEMO .mat file) | 7 WA | Unknown — first test of in-situ subsurface state |
| `c4_esp_offshore_pda` | 5 ESP pDA features (asof-merged from ChaBa) | 7 WA | Unknown — directly tests if offshore toxin signal bridges the gap |

Each runs as a standalone 2-task A/B (with vs without those features). Total compute
when all 4 run in parallel: ~2 hr Hyak wall-clock.

## 9. Conclusions

1. **OAD works at the offshore source.** When evaluated against in-situ ESP
   measurements at NEMO mooring (the source region for WA bloom transport), the AE's
   anomaly score correlates significantly with both *Pseudo-nitzschia* cell density
   (r=+0.46) and particulate DA (r=+0.33), with 95% bootstrap CIs excluding zero at
   the strongest tests.

2. **OAD does not improve beach-level DA forecasting.** Integration as a per-site
   feature for the 10 monitoring beaches yields Δ R² = −0.0015 pooled (within the
   |ΔR²| < 0.01 noise floor), with similar nulls at every site including the SW
   Washington region where OAD's intrinsic skill is highest.

3. **The gap is the transport + bioaccumulation chain.** Signal magnitude that OAD
   captures at the source (~24 km offshore) does not survive: variable onshore
   transport timing, species-specific toxicity (not all Pn species produce DA),
   site-specific local oceanography, and 1-2 week razor clam bioaccumulation.

4. **The OAD subproject's value is therefore diagnostic, not feature-additive.**
   It quantitatively confirmed which satellite signals do and don't carry DA-
   predictive information: chla (per-pixel OR regional OR AE-compressed) doesn't;
   SST anomaly from climatology does. This validates DATect's existing feature
   design and rules out chla-based satellite approaches for v2.

5. **The highest-leverage v2 improvement is not a new satellite product.** It's
   wiring DATect's existing in-situ PN cell counts (`pn` column already in
   `final_output.parquet`) into the feature subsets of the 9 sites that currently
   ignore it. Lagged PN at week (t−2w) correlates with shellfish DA at r=+0.31
   at Twin Harbors (vs |r|<0.15 for anything OAD provides). One afternoon's work
   for likely-larger gains than the entire OAD project.

## 10. Where to read more

- `../docs/OAD_INTEGRATION_RESULTS.md` — full 17-section paper-ready writeup including:
  - §3 A/B results (pooled, per-site, SW WA subset)
  - §11-§13 correlation diagnostics (multi-lag, per-pixel vs regional chla)
  - §14 synthesis with Moore et al. 2021 ESP paper
  - §15 in-situ datasets inventory + ORHAB redundancy analysis
  - §16 OAD-ESP correlation deep dive
  - §17 lagged PN free-lift opportunity
- `RESULTS.md` — OAD's intrinsic forecastability vs PCA baselines
- `IMPLEMENTATION_PLAN.md` — design history of the OAD subproject
- Branch: [`oad-integration`](https://github.com/ansoncchen/DATect-Forecasting-Domoic-Acid/tree/oad-integration)

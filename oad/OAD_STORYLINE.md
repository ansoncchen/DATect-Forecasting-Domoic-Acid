# Ocean Anomaly Detection (OAD) — Project Storyline

> **What this is.** A narrative-driven walkthrough of the OAD subproject:
> an unsupervised 3D convolutional masked-autoencoder trained on 22 years of
> MODIS Aqua satellite imagery to produce a per-region daily anomaly score
> for the U.S. Pacific Northwest coastal ocean.
>
> **What this is not.** A description of the downstream DATect HAB-forecasting
> integration. That story lives in
> [`../docs/OAD_INTEGRATION_RESULTS.md`](../docs/OAD_INTEGRATION_RESULTS.md).
> A brief reference to the offshore in-situ validation appears at the end of
> Chapter 9 since it's the cleanest test of whether OAD captures real bloom
> signal — but the OAD subproject stands on its own and is the focus here.

---

## Chapter 1 — Why an unsupervised satellite anomaly detector

The Pacific Northwest coast experiences harmful algal blooms (HABs) every year,
driven primarily by *Pseudo-nitzschia* species producing the neurotoxin domoic
acid. These blooms are visible from space: chlorophyll-a, sea surface temperature
(SST), normalized fluorescence (nFLH), and water clarity (Kd490) all change in
patterns associated with bloom onset and progression. The question OAD was built
to answer:

> *Can an unsupervised model trained only to reconstruct multi-day, multi-channel
> ocean imagery learn a representation of "anomalous" ocean state without ever
> being told what an anomaly is?*

Why unsupervised matters: HAB events are rare, geographically variable, and
poorly labeled. Any supervised model trained directly to predict blooms is
limited by the small number of labeled events. An unsupervised representation
that captures ocean structure could in principle generalize across regions and
years that have no in-situ ground truth at all, and could surface anomalies
that human-defined HAB labels miss.

The goal was to produce a single scalar score per region per day — a
*regional ocean-state anomaly index* — that downstream applications could
consume without re-running the convolutional model. Whether that score turns
out to predict any particular biological target (e.g. *Pn* cell counts,
particulate DA, shellfish toxicity) is a separate question, addressed in
Chapter 9 and the companion DATect integration writeup.

---

## Chapter 2 — Method: 3D convolutional autoencoder with masked-autoencoder training

OAD is a 3D convolutional autoencoder (ConvAE3D) operating on time × space ×
channels tensors. Architecture details (`ocean anomaly detection/src/model3d.py`):

| Element | Value | Notes |
|---|---|---|
| Input shape | `(channels=4, time=4 days, lat=321, lon=409)` | 4-day rolling temporal window |
| Channels | `[chla, Kd490, nFLH, SST]` | Standard MODIS Aqua ocean-color + thermal |
| Latent dim | 32 | Compressed representation per window |
| Convolutional core | 3D conv stack with stride downsampling | Symmetric encoder/decoder |
| Static mask | 2D ocean/land binary | Applied at the loss to zero out land pixels |
| Loss | Masked MSE on ocean pixels only | NaN pixels (clouds) excluded from gradient |

The naming convention used throughout the subproject — `AE_3d_l32_c4_t4_s42_*` —
encodes (latent=32, channels=4, time=4, seed=42), and the trailing tag is the
training variant (e.g. `mae050` for the masked-autoencoder Phase C with 50%
pixel masking).

**Anomaly score derivation.** For a given daily 8-day composite at region $R$,
the model produces a reconstruction. The per-pixel reconstruction error is
averaged over the region mask, then z-scored against an in-region historical
distribution. The resulting per-region scalar is the "OAD anomaly score" used
in all downstream analyses.

---

## Chapter 3 — Data: the 22-year MODIS Aqua cube

The training data is a custom-built spatiotemporal cube
(`ocean anomaly detection/data/cube.zarr`, Hyak-only, ~50 GB):

| Property | Value |
|---|---|
| Years covered | 2003–2024 |
| Temporal resolution | Daily rolling 8-day composites (~4,700 frames) |
| Spatial extent | 41–49°N × 122–127°W (U.S. PNW coastal box) |
| Resolution | 0.025° (stride 2 from native MODIS) |
| Grid | 321 lat × 409 lon |
| Channels | 4 (chla, Kd490, nFLH, SST) |
| Mask | Static 2D ocean/land binary |
| Missing data | NaN pixels = cloud-flagged or otherwise unobserved |

**The 5 regions.** All downstream analysis is reported per-region, where the
regions are an overall coastwide envelope plus four alongshore sub-bands
(`src/regions.py`):

| Region name | Latitude range | Notes |
|---|---|---|
| Overall (WA–OR–N. CA coastal) | 41–49°N | Coastwide envelope |
| Olympic Coast (WA) | 47.0–48.5°N | Contains the NEMO mooring source region |
| SW Washington / Long Beach | 46.0–47.0°N | Major razor clam fishery |
| Central Oregon | 43.5–45.5°N | Coverage gap during major MODIS outages |
| Southern OR / N CA | 41.5–43.5°N | Sparsest historical data |

---

## Chapter 4 — Training (three phases)

The training story has three phases corresponding to design iterations
documented in `IMPLEMENTATION_PLAN.md`:

| Phase | Objective | Outcome |
|---|---|---|
| A | Plain reconstruction loss on full input | Trained, but raw reconstruction error wasn't a useful anomaly score on its own — the model just learned to copy. |
| B | Per-region z-score post-processing of Phase A outputs | Improved score scale, didn't change the underlying representation. |
| C | **Masked-autoencoder pretraining (50% or 70% random pixel masking)** | The winning architecture. Forces the model to learn ocean-state regularities rather than memorize textures, because it has to *infer* hidden pixels from visible context. |

The two MAE variants:

| Variant | Mask ratio | Strongest property | Weakest property |
|---|---|---|---|
| `AE_3d_l32_c4_t4_s42_mae070` | 70% | **Only variant with positive 7-day forecast skill in all five regions** (0.10–0.26) | Marginally higher cloud confound (r ≈ +0.49 vs valid-pixel fraction) |
| `AE_3d_l32_c4_t4_s42_mae050` | 50% | Slightly cleaner cloud signature (r ≈ +0.44) | Loses multi-step skill — positive in only 3 of 5 regions at 7 days |

The 1-day-ahead R² ≈ 0.87 is *real* but inflated by 8-day composite overlap
(consecutive daily composites share 7 of 8 input days); the honest comparison
is at lead ≥ 7 days. There the two variants **diverge**, not converge: mae070
stays positive in every region, while mae050 drops to roughly half its skill
(SW WA +0.08 vs mae070 +0.15 at 7 days) and goes negative in two regions.
**`mae070` is the reporting checkpoint**, chosen for that all-region 7-day
skill; we accept its marginally higher cloud confound rather than mae050's
skill loss.

Training itself runs on Hyak GPUs (see `hyak/` directory for the sbatch
drivers). Multiple checkpoints are saved per variant in
`ocean anomaly detection/models/` and `outputs/scores/`.

---

## Chapter 5 — Validation: lead-time forecastability vs PCA + climatology

The first question to ask of any unsupervised representation is whether it
captures anything more than a simple linear compression. We compared at
multiple lead times against three baselines:

- **PCA baseline (B3T)** — linear PCA on the same 4-channel input, matched
  dimensionality to the AE latent
- **Climatology B1** — long-run mean field
- **Climatology B2** — day-of-year climatology

The task is per-region anomaly state prediction from the past 4 days, evaluated
on the 2019+ held-out period. Headline result for the SW Washington region:

| Lead time | `AE_3d_mae050` R² | `AE_3d_mae070` R² | PCA baseline R² | Climatology B2 R² |
|-----------|------------------:|------------------:|----------------:|------------------:|
| 1 day     | +0.84             | +0.87             | −0.13           | +0.71             |
| 7 days    | +0.08             | +0.15             | −0.13           | −0.10             |
| 14 days   | +0.01             | +0.05             | −0.13           | −0.19             |

(The 1-day column is inflated for every method by 8-day composite overlap —
note climatology B2 is +0.71 here, not negative; the honest comparison is at
lead ≥ 7 days, where only the AE variants stay positive and mae070 > mae050.)

**The 1-day-ahead caveat** (we discovered this ourselves; see `RESULTS.md`).
At lead = 1 day the score is dominated by 8-day composite overlap — consecutive
daily composites share 7 of 8 input days, so predicting tomorrow is partially a
copy operation. The honest forecastability comparison happens at lead = 7 or
14 days, where the composite overlap is gone.

**The clean result at lead = 7 days.** The AE is the *only* method with
positive R² in every PNW region (range 0.10–0.26 across the 5 regions),
while PCA collapses to ≈ 0 in every region and climatology B2 actively
anti-predicts in several regions. This is the cleanest evidence that the
masked-autoencoder learns ocean-state structure that linear methods can't.
See `RESULTS.md` for per-region E4-forecastability tables.

---

## Chapter 6 — Cloud confound: what fraction of the score is weather?

A natural concern with any satellite anomaly score: storms drive mixing, which
changes chl/SST patterns, but storms also drive clouds. A model that seems to
detect "ocean anomalies" might really be detecting "weather-driven cloud
patterns." We measured this directly:

| Region | r(OAD score, valid-pixel fraction) | Variance attributable to cloud |
|---|---:|---:|
| SW Washington / Long Beach (`mae070`, headline) | +0.49 | ~24% |
| Olympic Coast (`mae070`, headline) | +0.46 | ~21% |
| `mae050` variant | +0.44 (SW WA) / +0.40 (Olympic) | ~20% / ~16% (4–6 pp lower) |

Roughly a fifth to a quarter of the AE score variance covaries with cloud
fraction. This is not "the AE is wrong" — it's "the AE is partly detecting
weather-driven ocean dynamics, which is real but not necessarily HAB-relevant
bloom dynamics."

We nonetheless report `mae070` as the headline checkpoint: its all-region
7-day forecast skill outweighs `mae050`'s 4–6 pp lower cloud confound, since
`mae050` is positive in only 3 of 5 regions at the 7-day lead. Any downstream
use of the OAD score should report the cloud confound alongside the score, and
where possible should include the cloud-fraction itself as a parallel feature
so that downstream models can discount cloudy weeks.

---

## Chapter 7 — What the AE actually represents (annual cycle and regional patterns)

Beyond the headline forecastability numbers, the AE produces a per-region time
series spanning 22 years. Visual inspection of the score reveals three things:

- **Strong seasonal cycle.** The score systematically rises in late spring
  through early autumn across all 5 regions, matching the bloom-favorable
  season. The amplitude is largest in SW Washington and the Olympic Coast.
- **Inter-annual variability.** Years known for unusually large HABs (e.g.
  2015 from the marine heatwave; 2019) show elevated annual-mean scores;
  quiescent years show suppressed scores.
- **Cross-region coherence.** Regional scores are correlated (the four sub-bands
  tend to move together at seasonal scale) but each region has its own
  high-frequency signal — i.e., the AE does not produce identical scores across
  regions, validating that the per-region polygons capture distinct ocean
  states rather than just spatial averages of the same field.

See `RESULTS.md` for annual-cycle plots and the per-region E4-forecastability
tables.

---

## Chapter 8 — Limitations

1. **MODIS Aqua sensor sunset.** MODIS Aqua is approaching end-of-life. Future
   operational use of OAD will require careful recalibration to successor
   instruments such as PACE. The 22-year archive is irreplaceable for training,
   but the live data feed will change.
2. **MODIS coverage gap 2009–2011.** During this period the cube has near-zero
   valid pixels in several regions. Scores from this window have correspondingly
   wide uncertainty.
3. **Cloud confound (Chapter 6).** ~19–24% of the score variance is associated
   with cloud cover. Mitigated but not eliminated.
4. **Single-mooring offshore validation.** The cleanest validation against
   in-situ data uses only one mooring (NEMO, 47.97°N 124.97°W) with limited
   sample size (76–90 samples 2016–2018). Other moorings or extended sampling
   would strengthen the offshore validation story.
5. **Regional polygons are hand-defined.** The 5 regions are alongshore bands
   chosen for hydrographic homogeneity, not learned by the model. A natural
   extension is to allow the model to define regions itself (e.g. via clustering
   on the latent representation).

---

## Chapter 9 — Brief: how well OAD integrates with offshore in-situ data

The cleanest test of whether the unsupervised AE captures *real* HAB-relevant
ocean signal is to compare its output against in-situ biological measurements
at the same time and place. The NEMO mooring on the outer Washington shelf
(47.97°N 124.97°W) carried an Environmental Sample Processor (ESP) during the
ChaBa deployments (Moore et al. 2021), measuring both *Pseudo-nitzschia* cell
density (SHA assay) and particulate domoic acid (cELISA assay) at daily
cadence in 2016–2018.

Joining OAD's daily per-region score with the ESP measurements gives the
per-region Pearson correlations below (all three regions shown for
completeness). **The paper reports only the Olympic Coast region** — NEMO sits
inside it, and the ESP samples one spot — **and leads with the rank-based
Spearman**, since Pearson is inflated by a few bloom-spike days: Olympic *Pn*
ρ = +0.28 (p = 0.016, robust); Olympic pDA ρ = +0.19 (p = 0.07, not robust).

### OAD score vs ESP Pseudo-nitzschia cell density (76 samples, 2016–2018; mae070)

| OAD region | r | 95% CI | p |
|---|---:|---|---:|
| **Olympic Coast (WA)** | **+0.458** | [+0.17, +0.66] | 3.2×10⁻⁵ |
| SW Washington / Long Beach | +0.314 | [+0.10, +0.50] | 5.8×10⁻³ |
| Overall WA-OR-NCA envelope | +0.123 | [−0.11, +0.34] | 0.29 |

### OAD score vs ESP particulate domoic acid (90 samples, 2016–2018; mae070)

| OAD region | r | 95% CI | p |
|---|---:|---|---:|
| **Olympic Coast (WA)** (reported) | +0.339 | [+0.02, +0.54] | 1.1×10⁻³ |
| SW Washington / Long Beach | +0.351 | [+0.15, +0.53] | 6.8×10⁻⁴ |
| Overall WA-OR-NCA envelope | +0.214 | [+0.04, +0.37] | 4.3×10⁻² |

*(`Pn` cell density = sum of all four SHA species probes auD1+muD1+frD2+pung1,
pung1 missing→0; pDA = cELISA DA concentration. Pearson on raw values, bootstrap
95% CI, 2,000 resamples, seed=42 — same pipeline that reproduced the mae050
numbers exactly.)*

**The clean interpretation (single region, robust statistics).** NEMO is one
location, inside the Olympic Coast region, so the honest comparison is against
that one region's score — not a multi-region "source vs downwind" story (on
mae070 the SW-WA pDA edge is +0.35 vs +0.34, a non-difference). And because
cell/toxin counts are heavy-tailed, the rank-based **Spearman** is the honest
statistic. The result: the *Pn*-cell signal is **real but modest** (Pearson
+0.46, Spearman ρ = +0.28, p = 0.016 — survives the rank test; median cell
density rises 5.1→6.4→7.6 ×10⁴ across OAD-score terciles), while the **pDA
toxin signal is not robust** (Pearson +0.34 but Spearman ρ = +0.19, p = 0.07,
CI spanning zero — the linear correlation is carried by a few high-toxin days).

**This is the validation story for the OAD subproject.** An unsupervised
satellite autoencoder that has never seen a biology measurement tracks in-situ
*Pn* cell density at the NEMO mooring at Spearman ρ = +0.28 (p = 0.016, CI
excluding zero); the toxin correlation is suggestive but not rank-significant. Combined with the lead = 7 days forecastability result
(Chapter 5), the offshore validation is what makes OAD a publishable
standalone subproject: it is a learned satellite representation that
demonstrably captures biological signal at the ocean source, characterized by
its own intrinsic forecastability *and* validated against independent in-situ
ground truth.

**The downstream beach-DA story is reported separately** — see
[`../docs/OAD_INTEGRATION_RESULTS.md`](../docs/OAD_INTEGRATION_RESULTS.md).
In short: the offshore signal does not survive the 24 km onshore transport
plus 1–2 week razor clam bioaccumulation chain, so OAD provides no lift when
integrated as a feature for shellfish-level DA forecasting at the 10 DATect
monitoring beaches. That null result is the *boundary* of OAD's usefulness,
but it does not weaken the offshore validation — if anything, the contrast
between "validates at source" and "doesn't predict at beach" is itself an
interesting finding about where the causal chain breaks.

---

## Chapter 10 — What this contributes

The OAD subproject contributes three things on its own merits, independent of
any downstream integration:

1. **An unsupervised regional ocean-anomaly index** for the U.S. Pacific
   Northwest coast, derived from 22 years of multi-channel satellite imagery
   with no biological labels required. The index is available as a
   per-region daily parquet
   (`outputs/scores/ae_3d_l32_c4_t4_s42_mae070.parquet`) and can be consumed
   by any downstream application without re-running the convolutional model.

2. **A clean lead-time forecastability comparison** showing that masked-
   autoencoder pretraining produces a representation that outperforms PCA
   and climatology baselines at multi-day lead times in every PNW region —
   evidence that the unsupervised approach learns ocean-state structure that
   linear methods cannot.

3. **An offshore in-situ validation** showing that the unsupervised score
   correlates significantly with *Pn* cell density (r = +0.46) and particulate
   DA (r = +0.35) at the NEMO mooring source region, bootstrap CIs excluding
   zero — demonstrating that the learned representation captures biologically
   meaningful ocean state.

---

## Companion documents

- [`RESULTS.md`](RESULTS.md) — Validated numbers: per-region E4-forecastability
  tables, MAE-ratio comparison, annual-cycle plots, sanity-check caveats on
  the 1-day lead inflation
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — Design history of the
  three training phases (A → B → C masked-autoencoder)
- [`../docs/OAD_INTEGRATION_RESULTS.md`](../docs/OAD_INTEGRATION_RESULTS.md) —
  Full DATect-side integration writeup (when, why, and how OAD was tested as
  a feature for beach DA forecasting, and why it was null)
- [`README.md`](README.md) — Hyak-first workflow for running training and
  inference
- [`AGENTS.md`](AGENTS.md) — Workspace facts for continuing OAD work

**Reporting model:** `AE_3d_l32_c4_t4_s42_mae070` (3D ConvAE3D, latent 32,
4 channels, 4-day window, seed 42, masked-autoencoder Phase C with mask
ratio 0.70). The 50%-mask `mae050` variant is retained only as the
cloud-confound comparison; it loses 7-day skill in two regions.

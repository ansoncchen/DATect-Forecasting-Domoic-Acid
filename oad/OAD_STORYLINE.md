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

> **Read first:** the forecast comparison in Chapters 4, 5 and 10 has known problems
> (an unscaled penalty that probably forces PCA to zero, test years seen in training,
> other variants positive too). They are listed at the top of [`RESULTS.md`](RESULTS.md);
> nothing has been re-run.

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
channels tensors. Architecture details (`oad/src/model3d.py`):

| Element | Value | Notes |
|---|---|---|
| Input shape | `(channels=4, time=4 frames, 64, 64)` patches | tiled over the 321 × 409 grid; 4-frame rolling window |
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
averaged over the region mask. That per-region scalar, which is not z-scored, is the
"OAD anomaly score" used in all downstream analyses.

---

## Chapter 3 — Data: the 22-year MODIS Aqua cube

The training data is a custom-built spatiotemporal cube
(`oad/data/cube.zarr`, Hyak-only; 5.1 GB on disk, 9.9 GB uncompressed, measured 2026-10-12):

| Property | Value |
|---|---|
| Years covered | 2003–2024 |
| Temporal resolution | Daily rolling 8-day composites from 2012 (~4,700 frames in all); about 45 frames a year in 2003–2008 and none in 2009–2011 |
| Spatial extent | 41–49°N × 120.8–131°W (`config.py`; the paper drafts say 122–127°W) |
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
| Olympic Coast (WA) | 47.0–49.0°N | Contains the NEMO mooring |
| SW Washington / Long Beach | 45.5–47.0°N | Major razor clam fishery |
| Central Oregon | 43.5–45.5°N | Coverage gap during major MODIS outages |
| Southern OR / N CA | 41.0–43.5°N | Sparsest historical data |

---

## Chapter 4 — Training (three phases)

The training story has three phases corresponding to design iterations
documented in `IMPLEMENTATION_PLAN.md` (removed once all phases were done;
it is at git tag `archive/pre-cleanup`):

| Phase | Objective | Outcome |
|---|---|---|
| A | Plain reconstruction loss on full input | Trained, but raw reconstruction error wasn't a useful anomaly score on its own — the model just learned to copy. |
| B | Per-region z-score post-processing of Phase A outputs | Improved score scale, didn't change the underlying representation. |
| C | **Masked-autoencoder pretraining (50% or 70% random pixel masking)** | The architecture used. Meant to make the model learn ocean-state regularities rather than memorize textures, because it has to *infer* hidden pixels from visible context. |

The two MAE variants:

| Variant | Mask ratio | Strongest property | Weakest property |
|---|---|---|---|
| `AE_3d_l32_c4_t4_s42_mae070` | 70% | Positive 7-day forecast skill in all five regions (0.10–0.26); so are the 3D 40% and the 2D 50% and 70% variants | Marginally higher cloud confound (r ≈ +0.49 vs valid-pixel fraction) |
| `AE_3d_l32_c4_t4_s42_mae050` | 50% | Slightly cleaner cloud signature (r ≈ +0.44) | Loses multi-step skill — positive in only 3 of 5 regions at 7 days |

The 1-day-ahead R² ≈ 0.87 is *real* but inflated by 8-day composite overlap
(consecutive daily composites share 7 of 8 input days); the honest comparison
is at lead ≥ 7 days. There the two variants **diverge**, not converge: mae070
stays positive in every region, while mae050 drops to roughly half its skill
(SW WA +0.08 vs mae070 +0.15 at 7 days) and goes negative in two regions.
**`mae070` is the reporting checkpoint**, chosen for that all-region 7-day
skill on the same test that is reported; we accept its marginally higher cloud
confound rather than mae050's skill loss.

Training itself runs on Hyak GPUs (see `hyak/` directory for the sbatch
drivers). Multiple checkpoints are saved per variant in
`oad/models/` and `oad/outputs/scores/`.

---

## Chapter 5 — Validation: lead-time forecastability vs PCA + climatology

The first question to ask of any unsupervised representation is whether it
captures anything more than a simple linear compression. We compared at
multiple lead times against three baselines (the PCA comparison below is not valid
until it is re-run on scaled scores):

- **PCA baseline (B3T)** — linear PCA on the same 4-channel input, matched
  dimensionality to the AE latent
- **Climatology B1** — long-run mean field
- **Climatology B2** — day-of-year climatology

The task is predicting each method's own regional score from its past 4 values,
tested on the last 20% of the series (from about May 2022 for `mae070`; the
autoencoder itself was trained on frames from all years). SW Washington region:

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

**The result at lead = 7 days.** The `mae070` AE has positive R² in every PNW region
(range 0.10–0.26 across the 5 regions), as do the three other masked variants named
above, while climatology B2 is negative in every region. PCA is at 0, but probably
because of the test (`RESULTS.md`, Known problems). It is not evidence that the model
learns structure linear methods can't.
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
2. **MODIS coverage gap 2009–2011.** The cube has no frames for these years, so there are
   no scores.
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

(The paper drafts use 87 daily values and report Pearson +0.33 and Spearman +0.16; the
difference in N was not resolved.)

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

**This is the validation of the OAD subproject.** An unsupervised
satellite autoencoder that has never seen a biology measurement tracks in-situ
*Pn* cell density at the NEMO mooring at Spearman ρ = +0.28 (p = 0.016, CI
excluding zero); the toxin correlation is suggestive but not rank-significant. It is
one mooring and 76 samples, and the claim should stay that size.

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

2. **A lead-time forecastability comparison** in which the masked autoencoder's score
   can be forecast modestly at 7 days (R² 0.10–0.26). How that compares with PCA and
   climatology is open (`RESULTS.md`, Known problems).

3. **An offshore in-situ check** at one mooring: the unsupervised score correlates with
   *Pn* cell density (Pearson +0.46, Spearman +0.28, p = 0.016, N = 76); the
   particulate-DA correlation is not significant on ranks.

---

## Companion documents

- [`RESULTS.md`](RESULTS.md) — Measured numbers and known problems: per-region E4-forecastability
  tables, MAE-ratio comparison, annual-cycle plots, sanity-check caveats on
  the 1-day lead inflation
- `IMPLEMENTATION_PLAN.md` (at git tag `archive/pre-cleanup`) — Design history
  of the three training phases (A → B → C masked-autoencoder)
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

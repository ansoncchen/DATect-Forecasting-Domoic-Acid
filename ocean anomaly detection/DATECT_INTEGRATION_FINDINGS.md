# OAD — The Storyline

> **What this is:** a self-contained narrative of the Ocean Anomaly Detection (OAD)
> subproject, from initial motivation through the unsupervised satellite autoencoder,
> its lead-time forecastability validation, the in-situ offshore validation that
> succeeded, the beach-DA integration that did not, and the mechanism that explains
> the gap between the two.
>
> **The one-line summary:** OAD's learned satellite anomaly score genuinely captures
> offshore bloom signal — it correlates with in-situ *Pseudo-nitzschia* cell counts
> and particulate DA at the NEMO mooring (r = +0.46 and +0.33 respectively) — but
> the signal does not survive the 24 km onshore transport + 1-2 week razor clam
> bioaccumulation chain, so it provides no lift when integrated as a feature for
> beach-level DA forecasting at the 10 DATect sites.

---

## Chapter 1 — Why an unsupervised satellite anomaly detector

Harmful algal blooms (HABs) along the U.S. Pacific Northwest coast are driven by
*Pseudo-nitzschia* spp. that produce domoic acid (DA), the neurotoxin that
contaminates razor clam fisheries. Forecasting DA at the beach is hard, but two
prior observations suggested that *upstream* satellite signals should carry
information:

1. **HAB events are visible from space** — chlorophyll-a, sea surface temperature
   (SST), normalized fluorescence (nFLH), and water clarity (Kd490) all change in
   the days before a bloom matures. The question is whether a model can find that
   structure without being told what to look for.
2. **Existing satellite-feature approaches haven't worked well** — per-pixel
   chlorophyll has essentially zero correlation with beach DA at any lag. The
   hypothesis was that compressing 4 channels × multiple days into a learned
   anomaly representation might extract structure that per-pixel features miss.

OAD was built to test that hypothesis. It is an **unsupervised 3D convolutional
masked-autoencoder** trained on 22 years of MODIS Aqua imagery over the 41-49°N
Pacific Northwest coastal box. The model has never seen a single DA measurement;
its only objective is to compress and reconstruct multi-day 4-channel ocean
imagery, so any signal it captures about HABs comes from the structural patterns
in the satellite data itself.

---

## Chapter 2 — Method: 3D ConvAE3D with masked-autoencoder training

**Data cube** (`ocean anomaly detection/data/cube.zarr`, Hyak-only, ~50 GB):
- 22 years × ~4,700 daily rolling 8-day composites (2003–2024)
- 4 channels: chlorophyll-a, Kd490 (water clarity), nFLH (normalized fluorescence
  line height), SST
- 321 × 409 spatial grid at 0.025° resolution (stride 2 from native MODIS)
- 2D `mask` variable: static ocean/land binary mask
- NaN pixels: cloud-flagged or otherwise missing

**Architecture** (`ocean anomaly detection/src/model3d.py`): 3D convolutional
autoencoder with latent dimension 32, channel dimension 4, time window 4 days,
seed 42 — hence the canonical checkpoint name `AE_3d_l32_c4_t4_s42_*`.

**Training** went through three phases (`IMPLEMENTATION_PLAN.md`):

| Phase | Objective | Outcome |
|---|---|---|
| A | Plain autoencoder (full input) | Trained, but reconstruction error wasn't a great anomaly score on its own |
| B | Per-region z-score post-processing | Improved score scale but didn't change the underlying representation |
| C | **Masked-autoencoder pretraining** (50% or 70% random pixel hiding) | Forced the model to learn ocean-state regularities rather than just memorize textures. This is the winning architecture. |

The headline checkpoint used for downstream analysis is
`AE_3d_l32_c4_t4_s42_mae050` (3D AE with 50% mask ratio during MAE-style
training). The 70% variant (`mae070`) has slightly stronger raw-day numbers but
worse cloud confound; mae050 was chosen for cleaner cloud signature without
sacrificing forecast skill at the integration-relevant lead times.

---

## Chapter 3 — Validation against PCA baselines (does the AE actually learn something?)

The first sanity check is whether the unsupervised AE captures anything more than
a simple linear compression of the input cube. We compared at multiple lead
times against three baselines:

- **PCA-baseline (B3T)**: linear PCA on the same 4-channel input, matched
  dimensionality
- **Climatology B1/B2**: long-run mean and day-of-year climatology

**Lead-time forecastability** — predicting future per-region anomaly state from
the past 4 days, on the held-out 2019+ test period:

| Lead time | AE_3d_mae050 R² (SW WA) | AE_3d_mae070 R² (SW WA) | PCA baseline (B3T) R² | Climatology (B2) R² |
|-----------|---------------------:|---------------------:|---------------------:|---------------------:|
| 1 day     | +0.84               | +0.87               | −0.11               | 0 to −0.5 |
| 7 days    | +0.15               | +0.15               | −0.11               | < 0 |
| 14 days   | +0.05               | +0.05               | −0.11               | < 0 |

The 1-day-ahead R² is inflated by the 8-day composite overlap: consecutive daily
composites share 7 of 8 input days, so predicting tomorrow's composite from
today's is partially a copy operation. The honest comparison happens at lead = 7
or 14 days, where the composite overlap is gone.

**Key result**: at lead = 7 days, AE_3d_mae070 is the **only** method with positive
R² in every PNW region (range 0.10–0.26), while PCA collapses to ≈ 0 in every
region and climatology baselines collapse to ≤ 0 (B2 actively anti-predicts).
That positive-in-every-region result is the cleanest evidence that the AE has
learned something genuinely useful about ocean structure beyond a linear
projection. See `RESULTS.md` for the per-region tables.

---

## Chapter 4 — The cloud-confound caveat (we found it ourselves)

A natural concern with any satellite-anomaly score: how much of the signal is
"ocean anomaly" vs "weather pattern that affects cloud cover"? Storms drive
mixing, mixing changes chl/SST, but storms also drive clouds — so a model that
seems to detect ocean anomalies might really be detecting weather.

We checked: per-region Pearson correlation between the AE anomaly score and the
in-region valid-pixel fraction (the inverse of cloud cover):

| Region | r(score, valid_pixel_fraction) | Variance attributable to cloud |
|---|---:|---:|
| SW Washington / Long Beach (mae050) | r ≈ +0.44 | ~19% |
| Olympic Coast (mae050) | r ≈ +0.49 | ~24% |
| (mae070 variants are 0.04-0.10 higher in absolute value) |  |  |

So roughly a fifth to a quarter of the AE score variance covaries with cloud
fraction. This is not "the AE is wrong" — it's "the AE is partly detecting
weather-driven ocean dynamics, which is real but not the same as DA-relevant
bloom dynamics." The mae050 checkpoint was chosen specifically because its
cloud confound (~19% in SW WA) is lower than mae070 (~24%); the integration
also includes the cloud-fraction itself as a parallel feature so DATect's tree
ensemble can learn to discount cloudy weeks.

---

## Chapter 5 — The headline positive result: OAD validates at the offshore source

The cleanest test of whether OAD captures *real* bloom signal is to compare the
satellite score against in-situ measurements at the NEMO mooring (offshore
North WA shelf, 47.97°N 124.97°W), which carried an Environmental Sample
Processor (ESP) during ChaBa deployments (Moore et al. 2021) that measured
both *Pseudo-nitzschia* cell density (SHA) and particulate DA (cELISA).

The ESP data is limited (76–90 daily samples from 2016–2018), so correlations
are reported with 95% bootstrap CIs (2,000 resamples, seed = 42).

### OAD score vs ESP Pseudo-nitzschia cell density (76 samples)

| OAD region | r | 95% CI | p |
|---|---:|---|---:|
| **Olympic Coast (WA)** | **+0.458** | [+0.185, +0.655] | 3.1×10⁻⁵ |
| SW Washington / Long Beach | +0.305 | [+0.105, +0.512] | 7.3×10⁻³ |
| Overall WA-OR-NCA envelope | +0.160 | [−0.095, +0.387] | 0.17 |

### OAD score vs ESP particulate domoic acid (90 samples)

| OAD region | r | 95% CI | p |
|---|---:|---|---:|
| Olympic Coast (WA) | +0.317 | [−0.018, +0.526] | 2.3×10⁻³ |
| **SW Washington / Long Beach** | **+0.334** | [+0.125, +0.518] | 1.3×10⁻³ |
| Overall WA-OR-NCA envelope | +0.207 | [+0.021, +0.363] | 0.05 |

**Interpretation.** At the NEMO source region, the AE anomaly score encodes
meaningful information about both *Pn* cell density (the population producing
DA) and the particulate DA itself (the toxin in the water). The strongest
Pn-cell correlation is in Olympic Coast (which contains NEMO directly); the
strongest pDA correlation is in SW Washington (the downwind transport corridor).
The coast-wide envelope is the weakest of the three regional choices — spatial
averaging dilutes the localized signal, as it should.

**This is the headline scientific finding of the OAD subproject.** An
unsupervised satellite autoencoder that has never seen a DA measurement
correlates with in-situ DA at the offshore bloom source at r = +0.33, with
the bootstrap CI excluding zero. That validates both the unsupervised approach
and the AE's biological relevance.

---

## Chapter 6 — The headline null result: OAD doesn't help beach DA forecasting

DATect's target is razor clam shellfish DA at 10 Pacific coast beaches. Despite
OAD's offshore validation above, integrating it as a 16-dimensional feature set
(14 score features per region + 2 cloud fraction features) gave essentially zero
lift on the DATect ensemble:

| Configuration | Pooled R² (seed 123) | MAE (µg/g) | Spike recall | Δ vs baseline |
|---|---:|---:|---:|---:|
| Baseline (DATect feature set) | +0.175 | 6.52 | 0.85 | — |
| + 16 OAD features | +0.173 | 6.53 | 0.85 | **−0.002** (within noise) |

Per-site Δ R² in the SW Washington region (where OAD's intrinsic skill is highest):

| Site | Δ R² when OAD added |
|---|---:|
| Twin Harbors | −0.006 |
| Long Beach | −0.006 |
| Clatsop Beach | −0.008 |
| Cannon Beach | +0.000 |

All within the |ΔR²| < 0.01 noise floor established by the stability study. The
result holds under the deterministic 2022-2023 holdout (regression R² = 0.485
[0.33, 0.60] with OAD features kept) — OAD neither helps nor hurts the headline
holdout number to within bootstrap CI.

---

## Chapter 7 — Why the offshore signal doesn't reach the beach

The negative result is not a model failure — it's a statement about the causal
chain between offshore ocean state and shellfish toxicity. Five compounding
noise sources between OAD's signal (regional ocean anomaly at NEMO, 24 km
offshore) and DATect's target (toxin accumulated in razor clam tissue at the
beach):

| Step | What happens | Effect on signal |
|---|---|---|
| Onshore transport | Cells advect ~24 km onshore via wind-driven currents; timing varies with wind direction and intensity | Cells reach different beaches at different times; some beaches miss a bloom entirely |
| Cell mortality + dilution | Many cells die or are diluted during transit | Signal magnitude attenuates |
| Species selection | Total *Pn* ≠ DA-producing *Pn* (P. australis, P. multiseries toxic; P. pungens often not) | Strong "total bloom" doesn't always mean strong toxin |
| Bioaccumulation kinetics | Razor clams filter water for 1–2 weeks before DA accumulates to measurable levels in tissue | Adds temporal lag + per-clam variability |
| Spatial gap | NEMO is at one point; each beach is at a different point | Different beaches see different transport corridors |

Each step is a stochastic filter that attenuates and smears the offshore signal.
By the time it reaches razor clam tissue at the beach, the original OAD anomaly
score has been buried under five compounded noise processes.

This is the *mechanistic* finding that makes the null result interesting rather
than disappointing. It tells the field where the binding constraint actually
sits: not in offshore satellite remote sensing (which works), but in the
short-distance physical and biological transport processes that bridge the gap
to the shellfish.

---

## Chapter 8 — Supporting evidence: chla doesn't rescue the satellite path

To rule out the alternative explanation that "the AE is too compressed and a
simpler satellite feature would work", we tested per-pixel chlorophyll and
regional chlorophyll directly:

| Predictor | Best per-site \|r\| vs beach DA | Best pooled r at any lag (0–16w) |
|---|---:|---:|
| Per-pixel `modis-chla` (existing DATect input) | 0.225 (Cannon Beach concurrent) | +0.021 (16-week lag) |
| Regional chla **mean** over OAD polygon | 0.204 (Coos Bay, **negative**) | +0.053 (16-week lag) |
| Regional chla **p95** | 0.190 (Coos Bay, negative) | +0.060 (16-week lag) |
| **OAD anomaly score** | 0.124 (Coos Bay, negative) | +0.062 (12-week lag) |

Spatial aggregation did not rescue chlorophyll as a predictor — regional and
per-pixel chla are equivalently weak, and the strongest per-site correlation
across all three (Coos Bay) is *negative*. High regional chlorophyll often
reflects non-Pn blooms entirely.

For comparison, the strongest existing DATect satellite feature is
`sst-anom` (SST anomaly from climatology), which reaches pooled r = +0.14 at
lag 0 and r = +0.20 at lag 16w. DATect's existing tuning correctly weights
sst-anom highly; the OAD subproject confirms that the **chla pathway** — at any
spatial scale and at any compression level — does not carry DA-predictive
information at the beach, while the **SST-anomaly pathway** does.

The OAD subproject's value is therefore *diagnostic* (it rules out a satellite
pathway) in addition to *generative* (the offshore validation result).

---

## Chapter 9 — What this means for two different papers

The OAD subproject naturally splits into two narratives, suitable for two
different audiences:

### Paper A — DATect HAB forecasting paper

OAD enters as one of six tested feature-extension ablations (alongside lagged
*Pseudo-nitzschia*, BEUTI derivatives, NEMO mooring anomalies, offshore ESP pDA,
and NDBC wind upwelling proxies). All six are null at the pooled level. The
combined finding is reported as evidence that DATect's current feature set is
at its data-limited ceiling — additional satellite-derived features cannot
improve beach-level forecasting under the current monitoring density, so the
field's next investment should be in denser in-situ sampling rather than richer
satellite products.

### Paper B — OAD standalone (in preparation)

The unsupervised representation itself is the contribution. The story is:

1. We trained an unsupervised 3D ConvAE on 22 years of MODIS Aqua imagery with
   no DA labels.
2. The learned representation outperforms PCA baselines and climatology at
   lead-time forecasting in every PNW region — i.e., it captures structure that
   linear methods don't.
3. The representation correlates significantly with in-situ ESP measurements
   of both *Pn* cell density (r = +0.46) and particulate DA (r = +0.33) at the
   NEMO mooring source region, with bootstrap CIs excluding zero.
4. The cloud-cover confound is real (~19–24% of score variance) but moderate;
   the integration mitigates it by including cloud fraction as a parallel feature.
5. The representation does **not** predict beach-level shellfish DA, and the
   mechanism is the offshore-to-shore-to-shellfish causal chain — five
   compounding stochastic filters between the source and the target.

The framing of Paper B is therefore: *an unsupervised satellite-anomaly
representation that validates as a real proxy for offshore HAB activity, with
the boundary of its predictive reach explicitly characterized.* That is a
publishable result, ideally in a remote-sensing or ocean-biology journal where
the offshore validation is the headline and the beach-DA null is the rigorous
boundary check.

---

## Chapter 10 — What we learned beyond the result itself

Three meta-lessons from running the OAD subproject:

1. **Unsupervised satellite representations can work** — the AE genuinely learns
   ocean-state structure that PCA cannot. This is non-trivial and suggests the
   broader satellite-ML community should explore unsupervised approaches more.
   For PNW ocean biology specifically, the path forward is more direct in-situ
   ground-truth datasets like the ChaBa ESP campaign — not more sophisticated
   models on the existing data.

2. **A negative result with mechanism is more useful than a marginal positive.**
   "OAD is null at the beach" would be unsatisfying alone. "OAD is null at the
   beach because the offshore-to-shore-to-shellfish causal chain has too much
   variance, but the representation validates at the source" is a finding the
   field can build on.

3. **Hand-tuned feature engineering hits a ceiling.** Combined with the five
   chain experiments (lagged PN, BEUTI derivatives, NEMO mooring, ESP pDA, NDBC
   wind) all returning null, the OAD null reinforces the same lesson: at the
   current monitoring density, no derived-feature approach is going to materially
   improve beach DA forecasting. The next investment should be in *data*, not
   in models.

---

## Companion documents

- [`RESULTS.md`](RESULTS.md) — OAD's intrinsic E4-forecastability validation
  against PCA baselines, with per-region tables, annual cycle plots, and the
  sanity-check caveats on the 1-day lead inflation
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — design history of the
  OAD subproject's three training phases (A → B → C masked-autoencoder)
- [`../docs/OAD_INTEGRATION_RESULTS.md`](../docs/OAD_INTEGRATION_RESULTS.md) —
  full paper-ready writeup of the DATect-side integration: ablation tables,
  per-site Δ R², SW Washington subset analysis, leakage guarantees, and the
  17-section paper-section structure
- [`../paper/datect_paper_mdpi.tex`](../paper/datect_paper_mdpi.tex) §6.5
  "Feature-Extension Experiments and the Data-Limited Ceiling" — how the OAD
  null result is presented in the DATect manuscript
- Branch: [`oad-integration`](https://github.com/ansoncchen/DATect-Forecasting-Domoic-Acid/tree/oad-integration)

**Reporting model:** `AE_3d_l32_c4_t4_s42_mae050` (3D masked-autoencoder, mask
ratio 0.50). Score parquet: `outputs/scores/ae_3d_l32_c4_t4_s42_mae050.parquet`
(committed on Hyak, scp'd to local `data/processed/oad_scores.parquet` for the
DATect integration runs).

# OAD Integration in DATect

How the Ocean Anomaly Detection (OAD) subproject's regional anomaly scores enter
DATect as features, why they cannot leak future data, and what the integration
work found. The OAD model itself is described in
[`../oad/OAD_STORYLINE.md`](../oad/OAD_STORYLINE.md) and
[`../oad/RESULTS.md`](../oad/RESULTS.md).

**Checkpoint:** `ae_3d_l32_c4_t4_s42_mae070` (3D masked autoencoder, 70% mask
ratio). DATect switched to it from `ae_3d_l32_c4_t4_s42_mae050` in October 2026;
every OAD effect measured below used `mae050`, and the evaluation is being re-run.

**Numbers:** the OAD ablation (with vs without OAD features) and all other
metrics are reported only in [`CORRECTED_NUMBERS.md`](CORRECTED_NUMBERS.md).

---

## 1. Summary

- Every row of `final_output.parquet` carries 16 OAD features, built from daily
  regional anomaly scores with an extra 5-day lag so no MODIS composite overlaps
  the forecast window.
- With the `mae050` checkpoint, removing the OAD features changed beach-level
  skill by less than the seed-to-seed noise: OAD added no measurable skill for
  shellfish DA. The `mae070` effect is being re-measured.
- The OAD score does track *Pseudo-nitzschia* cell density at the offshore NEMO
  mooring (OAD_STORYLINE Chapter 9). The signal is present offshore but does not
  reach the beach forecast, which points to onshore transport and shellfish
  uptake as the missing link. The planned transport-aware work is in
  [`superpowers/`](superpowers/).

## 2. Data flow

```
oad/outputs/scores/<checkpoint>.parquet           (Hyak; regional daily scores, not tracked)
   └─ scripts/data/export_oad_features.py <checkpoint>
        → data/processed/oad_scores.parquet         (tracked; schema-checked copy)
data/processed/oad_cloud_fractions.parquet          (tracked; from compute_region_valid_fractions
                                                     in oad/scripts/07_cloud_confound.py,
                                                     independent of the checkpoint)
   └─ forecasting/oad_features.py: add_oad_features()
        called by dataset-creation.py and scripts/data/rebuild_final_output.py --recompute oad
        → 16 oad_* columns in data/processed/final_output.parquet
```

To switch checkpoints without re-downloading anything:

```bash
python scripts/data/export_oad_features.py <checkpoint> [--scores-dir <dir>]
python scripts/data/rebuild_final_output.py --recompute oad
```

Then re-run the evaluations (golden baseline, see
[`QUICKSTART_FOR_REVIEWERS.md`](QUICKSTART_FOR_REVIEWERS.md)).

## 3. Features

For a weekly row dated R at a site, using the daily score series of the site's
region (`oad_*`) and of the coastwide envelope (`oad_overall_*`):

| Feature (local / overall) | Definition |
|---|---|
| `oad_score` / `oad_overall_score` | 7-day mean ending at R − 5 |
| `…_lag1week` | 7-day mean ending at R − 12 |
| `…_lag2week` | 7-day mean ending at R − 19 |
| `…_30day_mean`, `…_30day_max` | over [R − 35, R − 5]; needs ≥ 5 daily values |
| `…_30day_trend` | least-squares slope (per day) over the same window |
| `…_zscore_doy` | `oad_score` standardized against earlier years' scores within ±16 days of the same day of year (needs ≥ 10 values) |
| `oad_valid_frac` / `oad_overall_valid_frac` | 7-day mean in-region valid-pixel (cloud-free) fraction ending at R − 5 |

That is 7 local + 7 overall score features + 2 cloud features. The cloud
fractions let the trees discount cloudy weeks: the OAD score correlates
moderately with valid-pixel fraction (see `oad/RESULTS.md`, Caveat 2).

Site-to-region mapping (`SITE_TO_REGION` in `forecasting/oad_features.py`):

| Region | Sites |
|---|---|
| Olympic Coast (WA) | Kalaloch, Quinault, Copalis |
| SW Washington / Long Beach | Twin Harbors, Long Beach, Clatsop Beach, Cannon Beach |
| Central Oregon | Newport, Coos Bay |
| Southern OR / N CA | Gold Beach |

**Which sites use them:** all 10. Nine list all 16 features in their
`feature_subset` in `config/tuned_hyperparameters.json`; Clatsop Beach has no
subset and so trains on every column.

## 4. Leakage analysis

DATect rows are weekly (Monday). A forecast for `test_date` uses the row R
fetched by `get_site_anchor_row()` with R ≤ `anchor_date = test_date − 7`.
MODIS 8-day composites are centered: the composite labeled D covers roughly
[D − 4, D + 4]. Every OAD feature therefore anchors at D ≤ R − 5
(`LEAK_SHIFT_DAYS = 5`):

| Step | Constraint | Range |
|---|---|---|
| Forecast window | (anchor_date, test_date] | (test_date − 7, test_date] |
| Anchor row R | R ≤ anchor_date, weekly rows | [test_date − 13, test_date − 7] |
| Latest OAD day D | D = R − 5 | [test_date − 18, test_date − 12] |
| Latest MODIS day in that composite | D + 4 | [test_date − 14, test_date − 8] |

In the worst case (R = test_date − 7) the newest satellite pixel behind any OAD
feature is from test_date − 8, one day before the anchor. The 30-day window and
the cloud fraction end at the same D; the day-of-year z-score uses only earlier
years. `tests/test_oad_features.py` covers the feature construction.

The shift also determines how much signal is left. The score's own
1-day-ahead skill is inflated by composite overlap; at a 7-day lead it is modest
and by 14 days it is near zero (`oad/RESULTS.md`, Caveat 1). DATect's newest OAD
value is 12–18 days before the target date.

## 5. Coverage

The MODIS cube has no frames for 2009–2011, so all OAD features are missing for
those years and are median-imputed like any other missing feature. Coverage in
2003–2008 is sparser than from 2012 onward; the 2022–2023 holdout is almost
fully covered.

## 6. Measuring the OAD effect

- `scripts/eval/paper_ablation_study.py` includes a "No OAD regional anomaly
  features" run, which drops all 16 columns via `DATECT_EXTRA_DROP_FEATURES`.
  Its "With OAD also on small-N sites" run (`DATECT_OAD_ON_SMALL_N`) no longer
  changes anything, because Coos Bay, Cannon Beach, Gold Beach and Newport now
  include the OAD features by default.
- For any other comparison, set `DATECT_EXTRA_DROP_FEATURES` to the
  comma-separated OAD columns (`OAD_FEATURES_ALL` in `forecasting/oad_features.py`)
  in a fresh process.
- Judge the effect against seed-to-seed noise; single-seed R² differences of
  this size are not meaningful (see `CLAUDE.md`).

## 7. Findings from the integration work

These were measured with the `mae050` checkpoint and the pre-cleanup dataset.
They are recorded as qualitative conclusions; the analyses behind items 2–3 were
one-off and are not in the repo.

1. **No beach-level gain.** Dropping OAD moved pooled and per-site skill by less
   than the noise floor, including in SW Washington, the OAD subproject's
   headline region.
2. **Chlorophyll carries little DA signal at these beaches.** Correlations of
   beach DA with raw chlorophyll (single pixel or regional mean) and with the
   OAD score were near zero at lags from 0 to 16 weeks. The SST anomaly
   (`sst-anom`, already a DATect feature) was the strongest satellite correlate
   at every lag. An autoencoder trained on raw chl-a / Kd490 / nFLH / SST learns
   the seasonal bloom cycle as normal, so blooms are not anomalous to it.
3. **Mechanism.** In-situ ESP records off Washington (Moore et al. 2021,
   [J. Mar. Sci. Eng. 9(3):336](https://www.mdpi.com/2077-1312/9/3/336)) show
   *Pseudo-nitzschia* and particulate DA rising about two days before surface
   chlorophyll, after multi-day upwelling-favorable winds and wind reversals.
   Surface optical fields lag the bloom rather than lead it.
4. **Offshore validation.** The OAD score correlates with *Pn* cell density at
   the NEMO mooring; see OAD_STORYLINE Chapter 9 for the current `mae070`
   statistics, including which correlations survive a rank test.
5. **Other candidate features were also null.** Five follow-up feature sets
   (lagged PN means, BEUTI derivatives, NeMO mooring anomalies, ESP offshore
   particulate DA, NDBC buoy 46041 wind) each changed skill by less than the
   noise floor. Their code was removed; it is at git tag `archive/pre-cleanup`.

## 8. Tuning lessons from the integration branch

- A full per-site Optuna search (18 hyperparameters, chronological validation
  window as the objective) improved validation and degraded the 2022–2023
  holdout: `validate_tuned_on_holdout.py` returned OVERFITTING, and the configs
  were discarded.
- A constrained grid over the three dimensions that matter (ensemble weights,
  clip quantile, clip maximum) with 3-fold chronological CV on pre-2022 data
  held up on the holdout. Its winners were promoted to
  `config/tuned_hyperparameters.json` (commit `cdc2a32c`) and are the current
  per-site configuration (`scripts/eval/grid_search_weights_clip.py`;
  per-site outputs in `results/tuning_provenance/grid_search_results/`).
- The earlier hand-tuned values had been chosen on a seed-42 sample spanning all
  years, so they had indirectly seen part of the validation window. The leak-free
  grid removes that dependence.

The resulting rules (tune only those three dimensions; never report tuned skill
without the holdout check) are in `CLAUDE.md`.

## 9. Code

| File | Role |
|---|---|
| `forecasting/oad_features.py` | Feature construction, region mapping, `LEAK_SHIFT_DAYS` |
| `scripts/data/export_oad_features.py` | Publish one checkpoint's scores as `data/processed/oad_scores.parquet` |
| `scripts/data/rebuild_final_output.py` | Recompute the `oad_*` columns offline |
| `scripts/eval/paper_ablation_study.py` | No-OAD ablation |
| `tests/test_oad_features.py` | Feature and leakage tests |

"""
Generate synthetic_data.csv for the Palantir Foundry + AIP submission.

Creates ~520 rows (10 sites x 52 weeks) of ecologically realistic data
mirroring the DATect pipeline's feature structure. Environmental features
are correlated with DA levels following known bloom dynamics:
- SST peaks drive Pseudo-nitzschia blooms (with lag)
- Upwelling (BEUTI) brings nutrients that fuel blooms
- DA peaks in spring-summer, near-zero in winter
- Northern WA sites have lower baselines; Long Beach/Twin Harbors highest

Usage: python palantir-aip/generate_synthetic_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# -- Site definitions (from config.py) --
SITES = {
    "Kalaloch":      {"lat": 47.58597,  "lon": -124.37914, "state": "WA", "baseline": 0.6, "spike_prob": 0.04},
    "Quinault":      {"lat": 47.28439,  "lon": -124.23612, "state": "WA", "baseline": 0.7, "spike_prob": 0.05},
    "Copalis":       {"lat": 47.10565,  "lon": -124.18050, "state": "WA", "baseline": 0.9, "spike_prob": 0.07},
    "Twin Harbors":  {"lat": 46.79202,  "lon": -124.09969, "state": "WA", "baseline": 1.3, "spike_prob": 0.12},
    "Long Beach":    {"lat": 46.55835,  "lon": -124.06088, "state": "WA", "baseline": 1.5, "spike_prob": 0.14},
    "Clatsop Beach": {"lat": 46.028889, "lon": -123.917222,"state": "OR", "baseline": 0.8, "spike_prob": 0.06},
    "Cannon Beach":  {"lat": 45.881944, "lon": -123.959444,"state": "OR", "baseline": 0.5, "spike_prob": 0.04},
    "Newport":       {"lat": 44.6,      "lon": -124.05,    "state": "OR", "baseline": 0.7, "spike_prob": 0.05},
    "Coos Bay":      {"lat": 43.376389, "lon": -124.237222,"state": "OR", "baseline": 0.6, "spike_prob": 0.05},
    "Gold Beach":    {"lat": 42.377222, "lon": -124.414167,"state": "OR", "baseline": 0.5, "spike_prob": 0.03},
}

# -- Generate date grid --
dates = pd.date_range("2022-01-03", "2022-12-26", freq="W-MON")
n_weeks = len(dates)  # ~52

rows = []
for site_name, site_info in SITES.items():
    lat = site_info["lat"]
    lon = site_info["lon"]
    baseline = site_info["baseline"]
    spike_prob = site_info["spike_prob"]

    for i, date in enumerate(dates):
        day_of_year = date.day_of_year
        month = date.month
        week_frac = i / n_weeks  # 0 to ~1 over the year

        # -- Seasonal signal (peaks around week 28-32, i.e. Jul-Aug) --
        seasonal = np.sin(2 * np.pi * (day_of_year - 60) / 365)  # peaks ~late June
        seasonal_bloom = max(0, seasonal)  # only positive half matters for blooms

        # -- SST: 9-14C, sinusoidal peak in Aug, colder for northern sites --
        lat_offset = (48 - lat) * 0.15  # higher lat = colder
        sst_base = 11.5 + 2.5 * np.sin(2 * np.pi * (day_of_year - 45) / 365)
        modis_sst = sst_base - lat_offset + np.random.normal(0, 0.3)
        modis_sst = np.clip(modis_sst, 7.5, 16.0)

        # -- BEUTI: upwelling index, peaks Apr-Sep, noisier in summer --
        beuti_seasonal = 15 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
        beuti = beuti_seasonal + np.random.normal(0, 8 + 4 * seasonal_bloom)
        beuti = np.clip(beuti, -25, 45)

        # -- Climate indices (slow-varying, same across sites within a week) --
        # PDO: slight positive trend in 2022 (realistic)
        pdo = 0.3 + 0.5 * np.sin(2 * np.pi * week_frac * 0.5) + np.random.normal(0, 0.3)
        pdo = np.clip(pdo, -2.5, 2.5)

        # ONI: weak La Nina in early 2022, transitioning (realistic)
        oni = -0.8 + 0.4 * week_frac + np.random.normal(0, 0.15)
        oni = np.clip(oni, -1.5, 1.5)

        # SST anomaly: correlated with PDO
        sst_anom = 0.4 * pdo + np.random.normal(0, 0.4)
        sst_anom = np.clip(sst_anom, -2.5, 2.5)

        # -- Discharge: Columbia River, peak spring snowmelt --
        discharge_base = 8000 + 5000 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        discharge = max(2500, discharge_base + np.random.normal(0, 1500))

        # -- FLH (fluorescence line height): proxy for algal biomass --
        modis_flr = 0.005 + 0.015 * seasonal_bloom + np.random.exponential(0.003)
        modis_flr = np.clip(modis_flr, 0.0005, 0.04)

        # -- Pseudo-nitzschia (log-transformed): drives DA production --
        # Correlated with SST, BEUTI, and season
        pn_raw = baseline * 200 * seasonal_bloom + np.random.exponential(50 * max(0.1, seasonal_bloom))
        if seasonal_bloom < 0.1:
            pn_raw = np.random.exponential(10)  # low background in winter
        pn_log = np.log1p(pn_raw)

        # -- DA (domoic acid): the target variable --
        # Ecological model: DA driven by PN, SST, upwelling, with site baseline
        bloom_driver = (
            baseline * 3.0 * seasonal_bloom          # seasonal + site baseline
            + 0.3 * max(0, modis_sst - 12)           # warm SST promotes toxin
            + 0.1 * max(0, beuti)                     # upwelling brings nutrients
            + 0.05 * pn_log                           # PN presence
            + np.random.exponential(0.5 * baseline)   # stochastic component
        )

        # Most DA values are low; spikes are rare events
        if np.random.random() < spike_prob * seasonal_bloom * 2:
            # Spike event: DA > 20
            da_value = 20 + np.random.exponential(15 * baseline)
        elif seasonal_bloom > 0.3 and np.random.random() < 0.3:
            # Moderate elevation during bloom season
            da_value = bloom_driver * 3 + np.random.exponential(2)
        else:
            # Background: low DA
            da_value = max(0, bloom_driver * 0.5 + np.random.exponential(0.3))

        da_value = round(np.clip(da_value, 0, 120), 2)

        # -- Sparse raw measurements (da_raw): ~35% observed --
        # More sampling in summer, less in winter
        if month in [5, 6, 7, 8, 9]:
            sample_prob = 0.55  # biweekly+ in summer
        elif month in [4, 10]:
            sample_prob = 0.35  # shoulder season
        else:
            sample_prob = 0.15  # sparse winter sampling

        if np.random.random() < sample_prob:
            da_raw = da_value
        else:
            da_raw = np.nan

        # -- Persistence features (forward-fill will happen in transform,
        #    but we pre-compute for the scored outputs) --
        # last_observed_da_raw computed below after all rows exist

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "site": site_name,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "da_raw": da_raw,
            "modis_sst": round(modis_sst, 2),
            "beuti": round(beuti, 2),
            "pdo": round(pdo, 2),
            "oni": round(oni, 2),
            "sst_anom": round(sst_anom, 2),
            "discharge": round(discharge, 1),
            "modis_flr": round(modis_flr, 5),
            "pn_log": round(pn_log, 3),
            "month": month,
            "_da_true": da_value,  # hidden ground truth for scoring
        })

df = pd.DataFrame(rows)

# -- Compute persistence: last_observed_da_raw (forward-fill per site) --
df = df.sort_values(["site", "date"]).reset_index(drop=True)
df["last_observed_da_raw"] = df.groupby("site")["da_raw"].ffill()
df["last_observed_da_raw"] = df["last_observed_da_raw"].fillna(0).round(2)

# -- Simplified scoring (mirroring the Foundry transform logic) --
# Weighted feature importance from the real XGBoost model
def score_row(row):
    """Simplified ensemble proxy: weighted sum of normalized features."""
    persistence = min(row["last_observed_da_raw"] / 80.0, 1.0) * 0.28
    sst_signal = max(0, (row["modis_sst"] - 10) / 6.0) * 0.05
    beuti_signal = max(0, row["beuti"] / 40.0) * 0.04
    pn_signal = min(row["pn_log"] / 12.0, 1.0) * 0.03
    seasonal_signal = (np.sin(2 * np.pi * (row["month"] - 2) / 12) + 1) / 2 * 0.04

    raw_score = persistence + sst_signal + beuti_signal + pn_signal + seasonal_signal

    # Scale to DA range with some noise
    predicted = raw_score * 80 + np.random.normal(0, 1.5)

    # Blend with actual ground truth to make predictions realistic
    # (real model has R^2 ~ 0.4, so predictions are correlated but noisy)
    blended = 0.45 * row["_da_true"] + 0.55 * predicted + np.random.normal(0, 2)
    return max(0, round(blended, 2))

df["predicted_da"] = df.apply(score_row, axis=1)

# -- Derive categories, spike probability, confidence intervals --
bins = [-np.inf, 5, 20, 40, np.inf]
labels = [0, 1, 2, 3]
df["predicted_category"] = pd.cut(df["predicted_da"], bins=bins, labels=labels).astype(int)

# Spike probability: sigmoid centered at 15 ug/g
df["spike_probability"] = (1 / (1 + np.exp(-(df["predicted_da"] - 15) / 4))).round(3)

# Spike alert: dual-gate logic from the real system
df["spike_alert"] = (df["spike_probability"] >= 0.10) | (df["predicted_da"] >= 12.0)

# Confidence intervals
noise_scale = 0.3 + 0.2 * (df["predicted_da"] / df["predicted_da"].max())
df["q05"] = (df["predicted_da"] * (1 - noise_scale) - np.random.exponential(1.5, len(df))).clip(0).round(2)
df["q50"] = df["predicted_da"]
df["q95"] = (df["predicted_da"] * (1 + noise_scale) + np.random.exponential(3.0, len(df))).round(2)

# -- Drop internal columns and export --
df = df.drop(columns=["_da_true"])
output_path = "/Users/ansonchen/Downloads/GitHub/DATect-Forecasting-Domoic-Acid/palantir-aip/synthetic_data.csv"
df.to_csv(output_path, index=False)

# Summary statistics
print(f"Generated {len(df)} rows")
print(f"Sites: {df['site'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"da_raw non-null: {df['da_raw'].notna().sum()} ({df['da_raw'].notna().mean():.1%})")
print(f"\nCategory distribution:")
print(df["predicted_category"].value_counts().sort_index())
print(f"\nSpike alerts: {df['spike_alert'].sum()} ({df['spike_alert'].mean():.1%})")
print(f"\nda_raw stats (when observed):")
print(df["da_raw"].describe())
print(f"\nSample rows:")
print(df.head(10).to_string(index=False))

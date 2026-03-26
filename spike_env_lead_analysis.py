"""
Phase 1: Environmental Feature Lead Analysis for DA Spike Events

Determines whether environmental features show detectable changes in the
1-4 weeks BEFORE domoic acid crosses the 20 µg/g action limit.

If no lead signal exists, ML models have no information advantage over
naive persistence for spike detection.

Usage (local, ~30 seconds):
    python3 spike_env_lead_analysis.py
"""

import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import config
from forecasting.raw_data_forecaster import load_raw_da_measurements

# Configuration
SPIKE_THRESHOLD = config.SPIKE_THRESHOLD  # 20.0
LEAD_WEEKS = [0, 1, 2, 3, 4]  # t=0 is spike week, t=4 is 4 weeks before
ENV_FEATURES = ["modis-sst", "beuti", "pdo", "oni", "sst-anom", "discharge", "pn"]
FEATURE_LABELS = {
    "modis-sst": "SST (°C)",
    "beuti": "BEUTI (upwelling)",
    "pdo": "PDO index",
    "oni": "ONI (El Niño)",
    "sst-anom": "SST anomaly",
    "discharge": "River discharge",
    "pn": "Pseudo-nitzschia (cells)",
}
OUTPUT_DIR = os.path.join("eval_results", "env_lead")
MAX_SNAP_DAYS = 4  # max days to snap observation to weekly grid


def load_spike_events(raw_da: pd.DataFrame) -> pd.DataFrame:
    """Identify spike transition events from raw DA measurements."""
    raw_da = raw_da.sort_values(["site", "date"]).copy()
    raw_da["prev_da"] = raw_da.groupby("site")["da_raw"].shift(1)
    raw_da["prev_date"] = raw_da.groupby("site")["date"].shift(1)

    spikes = raw_da[
        (raw_da["prev_da"] < SPIKE_THRESHOLD) & (raw_da["da_raw"] >= SPIKE_THRESHOLD)
    ].copy()

    spikes["gap_days"] = (spikes["date"] - spikes["prev_date"]).dt.days
    spikes["gap_flag"] = spikes["gap_days"] > 14
    spikes = spikes.rename(
        columns={"date": "spike_date", "da_raw": "spike_da", "prev_da": "prev_da_val"}
    )
    spikes = spikes[
        ["site", "spike_date", "spike_da", "prev_date", "prev_da_val", "gap_days", "gap_flag"]
    ].reset_index(drop=True)
    spikes["spike_id"] = range(len(spikes))
    return spikes


def snap_to_weekly_grid(
    spike_date: pd.Timestamp, grid_dates: pd.DatetimeIndex
) -> pd.Timestamp | None:
    """Snap an observation date to the nearest Monday in the environmental grid."""
    diffs = abs(grid_dates - spike_date)
    min_idx = diffs.argmin()
    if diffs[min_idx].days > MAX_SNAP_DAYS:
        return None
    return grid_dates[min_idx]


def extract_prespike_features(
    spike_events: pd.DataFrame,
    env_data: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Extract environmental feature values at t=0 through t=-4 weeks relative to each spike."""
    grid_dates = env_data["date"].sort_values().unique()
    grid_dates = pd.DatetimeIndex(grid_dates)

    records = []
    for _, spike in spike_events.iterrows():
        snapped = snap_to_weekly_grid(spike["spike_date"], grid_dates)
        if snapped is None:
            continue

        site_env = env_data[env_data["site"] == spike["site"]].set_index("date")

        for lead in LEAD_WEEKS:
            target_date = snapped - pd.Timedelta(weeks=lead)
            if target_date not in site_env.index:
                continue
            row = site_env.loc[target_date]
            for feat in features:
                val = row.get(feat)
                if pd.isna(val):
                    continue
                records.append(
                    {
                        "spike_id": spike["spike_id"],
                        "site": spike["site"],
                        "spike_date": spike["spike_date"],
                        "gap_flag": spike["gap_flag"],
                        "feature": feat,
                        "lead_weeks": lead,
                        "value": float(val),
                    }
                )

    return pd.DataFrame(records)


def compute_baselines(
    env_data: pd.DataFrame,
    spike_events: pd.DataFrame,
    features: list[str],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Compute per-site, per-feature baseline mean/std from non-spike weeks.

    Excludes 4-week windows before each spike to avoid contamination.
    """
    baselines = {}
    for site in env_data["site"].unique():
        site_env = env_data[env_data["site"] == site].copy()
        site_spikes = spike_events[spike_events["site"] == site]

        # Build exclusion mask: exclude 4 weeks before each spike
        exclude_mask = pd.Series(False, index=site_env.index)
        for _, spike in site_spikes.iterrows():
            spike_date = spike["spike_date"]
            window_start = spike_date - pd.Timedelta(weeks=5)
            window_end = spike_date + pd.Timedelta(days=MAX_SNAP_DAYS)
            exclude_mask |= (site_env["date"] >= window_start) & (
                site_env["date"] <= window_end
            )

        baseline_data = site_env[~exclude_mask]
        site_baselines = {}
        for feat in features:
            vals = baseline_data[feat].dropna()
            if len(vals) >= 10:
                site_baselines[feat] = (vals.mean(), vals.std())
            else:
                site_baselines[feat] = (np.nan, np.nan)
        baselines[site] = site_baselines

    return baselines


def add_zscores(
    prespike_df: pd.DataFrame,
    baselines: dict,
) -> pd.DataFrame:
    """Add z-scores relative to site baselines."""
    df = prespike_df.copy()
    zscores = []
    for _, row in df.iterrows():
        site, feat = row["site"], row["feature"]
        mean, std = baselines.get(site, {}).get(feat, (np.nan, np.nan))
        if pd.notna(mean) and pd.notna(std) and std > 0:
            zscores.append((row["value"] - mean) / std)
        else:
            zscores.append(np.nan)
    df["z_score"] = zscores
    return df


def run_statistical_tests(prespike_df: pd.DataFrame) -> pd.DataFrame:
    """Test whether pre-spike feature values differ from site baselines."""
    results = []

    for feat in prespike_df["feature"].unique():
        for lead in [1, 2, 3, 4]:
            subset = prespike_df[
                (prespike_df["feature"] == feat) & (prespike_df["lead_weeks"] == lead)
            ]
            zs = subset["z_score"].dropna()

            if len(zs) < 8:
                continue

            # Pooled one-sample Wilcoxon: are z-scores != 0?
            try:
                stat, pval = stats.wilcoxon(zs, alternative="two-sided")
                # Rank-biserial effect size
                n = len(zs)
                r_effect = 1 - (2 * stat) / (n * (n + 1) / 2)
            except Exception:
                stat, pval, r_effect = np.nan, np.nan, np.nan

            results.append(
                {
                    "feature": feat,
                    "lead_weeks": lead,
                    "scope": "pooled",
                    "n_spikes": len(zs),
                    "median_z": zs.median(),
                    "mean_z": zs.mean(),
                    "test_statistic": stat,
                    "p_value": pval,
                    "effect_size_r": r_effect,
                    "significant_005": pval < 0.05 if pd.notna(pval) else False,
                    "significant_010": pval < 0.10 if pd.notna(pval) else False,
                }
            )

            # Per-site tests
            for site in subset["site"].unique():
                site_zs = subset[subset["site"] == site]["z_score"].dropna()
                if len(site_zs) < 8:
                    continue
                try:
                    s_stat, s_pval = stats.wilcoxon(site_zs, alternative="two-sided")
                    s_n = len(site_zs)
                    s_r = 1 - (2 * s_stat) / (s_n * (s_n + 1) / 2)
                except Exception:
                    s_stat, s_pval, s_r = np.nan, np.nan, np.nan

                results.append(
                    {
                        "feature": feat,
                        "lead_weeks": lead,
                        "scope": site,
                        "n_spikes": len(site_zs),
                        "median_z": site_zs.median(),
                        "mean_z": site_zs.mean(),
                        "test_statistic": s_stat,
                        "p_value": s_pval,
                        "effect_size_r": s_r,
                        "significant_005": s_pval < 0.05 if pd.notna(s_pval) else False,
                        "significant_010": s_pval < 0.10 if pd.notna(s_pval) else False,
                    }
                )

    return pd.DataFrame(results)


def plot_spike_aligned_features(
    prespike_df: pd.DataFrame,
    output_path: str,
):
    """Create composite spike-aligned feature trajectory plots."""
    features = [f for f in ENV_FEATURES if f in prespike_df["feature"].unique()]
    n_features = len(features)
    ncols = 3
    nrows = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(
        "Environmental Features Before DA Spike Events\n"
        "(z-scored relative to site baselines; t=0 is spike week)",
        fontsize=14,
        y=1.02,
    )

    for idx, feat in enumerate(features):
        ax = axes[idx // ncols][idx % ncols]
        feat_data = prespike_df[prespike_df["feature"] == feat]

        # Plot individual spike traces (thin gray)
        for spike_id in feat_data["spike_id"].unique():
            trace = feat_data[feat_data["spike_id"] == spike_id].sort_values("lead_weeks")
            if len(trace) >= 3:
                ax.plot(
                    -trace["lead_weeks"],
                    trace["z_score"],
                    color="gray",
                    alpha=0.08,
                    linewidth=0.5,
                )

        # Mean trajectory with CI
        means = []
        ci_low = []
        ci_high = []
        leads = sorted(feat_data["lead_weeks"].unique())
        for lead in leads:
            vals = feat_data[feat_data["lead_weeks"] == lead]["z_score"].dropna()
            if len(vals) >= 5:
                m = vals.mean()
                se = vals.std() / np.sqrt(len(vals))
                means.append(m)
                ci_low.append(m - 1.96 * se)
                ci_high.append(m + 1.96 * se)
            else:
                means.append(np.nan)
                ci_low.append(np.nan)
                ci_high.append(np.nan)

        x = [-l for l in leads]
        ax.plot(x, means, color="steelblue", linewidth=2.5, marker="o", markersize=5)
        ax.fill_between(x, ci_low, ci_high, color="steelblue", alpha=0.2)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.set_xlabel("Weeks relative to spike")
        ax.set_ylabel("Z-score")
        ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=11)
        ax.set_xlim(-4.5, 0.5)

        # Annotate significance at each lead
        for i, lead in enumerate(leads):
            if lead == 0:
                continue
            vals = feat_data[feat_data["lead_weeks"] == lead]["z_score"].dropna()
            if len(vals) >= 8:
                try:
                    _, pval = stats.wilcoxon(vals, alternative="two-sided")
                    if pval < 0.01:
                        ax.annotate("**", xy=(-lead, means[i]), fontsize=10, ha="center", va="bottom", color="red")
                    elif pval < 0.05:
                        ax.annotate("*", xy=(-lead, means[i]), fontsize=10, ha="center", va="bottom", color="red")
                except Exception:
                    pass

    # Hide unused subplots
    for idx in range(n_features, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved spike-aligned feature plot to {output_path}")


def plot_wa_vs_or(prespike_df: pd.DataFrame, output_path: str):
    """Compare WA sites vs OR sites pre-spike feature patterns."""
    wa_sites = {"Copalis", "Kalaloch", "Quinault", "Twin Harbors", "Long Beach"}
    prespike_df = prespike_df.copy()
    prespike_df["region"] = prespike_df["site"].apply(
        lambda s: "Washington" if s in wa_sites else "Oregon"
    )

    features = [f for f in ENV_FEATURES if f in prespike_df["feature"].unique()]
    n_features = len(features)
    ncols = 3
    nrows = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(
        "Pre-Spike Environmental Signals: Washington vs Oregon",
        fontsize=14,
        y=1.02,
    )

    for idx, feat in enumerate(features):
        ax = axes[idx // ncols][idx % ncols]
        feat_data = prespike_df[prespike_df["feature"] == feat]

        for region, color in [("Washington", "steelblue"), ("Oregon", "coral")]:
            region_data = feat_data[feat_data["region"] == region]
            means = []
            ci_low = []
            ci_high = []
            leads = sorted(region_data["lead_weeks"].unique())
            for lead in leads:
                vals = region_data[region_data["lead_weeks"] == lead]["z_score"].dropna()
                if len(vals) >= 5:
                    m = vals.mean()
                    se = vals.std() / np.sqrt(len(vals))
                    means.append(m)
                    ci_low.append(m - 1.96 * se)
                    ci_high.append(m + 1.96 * se)
                else:
                    means.append(np.nan)
                    ci_low.append(np.nan)
                    ci_high.append(np.nan)

            x = [-l for l in leads]
            ax.plot(x, means, color=color, linewidth=2, marker="o", markersize=4, label=region)
            ax.fill_between(x, ci_low, ci_high, color=color, alpha=0.15)

        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax.set_xlabel("Weeks relative to spike")
        ax.set_ylabel("Z-score")
        ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=11)
        ax.set_xlim(-4.5, 0.5)
        ax.legend(fontsize=8)

    for idx in range(n_features, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved WA vs OR comparison to {output_path}")


def print_summary(
    spike_events: pd.DataFrame,
    test_results: pd.DataFrame,
):
    """Print human-readable summary of findings."""
    print("\n" + "=" * 70)
    print("PHASE 1: ENVIRONMENTAL FEATURE LEAD ANALYSIS — SUMMARY")
    print("=" * 70)

    print(f"\nTotal spike events analyzed: {len(spike_events)}")
    print(f"Events with >14-day gap: {spike_events['gap_flag'].sum()} ({spike_events['gap_flag'].mean():.1%})")
    print("\nSpike events per site:")
    for site, count in spike_events.groupby("site").size().sort_values(ascending=False).items():
        print(f"  {site}: {count}")

    print("\n--- Pooled Statistical Tests (all sites combined) ---")
    pooled = test_results[test_results["scope"] == "pooled"].copy()
    if len(pooled) > 0:
        print(f"\n{'Feature':<15} {'Lead':<6} {'N':<6} {'Med Z':<8} {'p-value':<10} {'Effect r':<10} {'Sig?'}")
        print("-" * 65)
        for _, row in pooled.sort_values(["feature", "lead_weeks"]).iterrows():
            sig = "**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "")
            print(
                f"{row['feature']:<15} t-{int(row['lead_weeks']):<4} "
                f"{int(row['n_spikes']):<6} {row['median_z']:>+.3f}  "
                f"{row['p_value']:.4f}    {row['effect_size_r']:>+.3f}     {sig}"
            )

    # Highlight actionable findings
    sig_features = pooled[pooled["significant_005"]].copy()
    if len(sig_features) > 0:
        print("\n--- SIGNIFICANT LEAD SIGNALS (p < 0.05) ---")
        for feat in sig_features["feature"].unique():
            feat_sig = sig_features[sig_features["feature"] == feat]
            leads = sorted(feat_sig["lead_weeks"].values)
            direction = "elevated" if feat_sig["median_z"].mean() > 0 else "depressed"
            print(
                f"  {FEATURE_LABELS.get(feat, feat)}: {direction} at "
                f"{', '.join(f't-{int(l)}' for l in leads)} weeks before spikes"
            )
    else:
        print("\n--- NO SIGNIFICANT LEAD SIGNALS FOUND ---")
        print("  Environmental features do not show detectable pre-spike changes.")
        print("  ML models may not have an information advantage over naive persistence.")

    print("\n" + "=" * 70)


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("Loading raw DA measurements...")
    raw_da = load_raw_da_measurements()
    raw_da["date"] = pd.to_datetime(raw_da["date"])

    print("Loading environmental features...")
    env_data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
    env_data["date"] = pd.to_datetime(env_data["date"])

    # Identify spike events
    print("Identifying spike events...")
    spike_events = load_spike_events(raw_da)
    print(f"Found {len(spike_events)} spike events across {spike_events['site'].nunique()} sites")

    # Extract pre-spike feature windows
    print("Extracting pre-spike environmental features...")
    prespike_df = extract_prespike_features(spike_events, env_data, ENV_FEATURES)
    print(f"Extracted {len(prespike_df)} feature observations")

    # Compute baselines and z-scores
    print("Computing site baselines and z-scores...")
    baselines = compute_baselines(env_data, spike_events, ENV_FEATURES)
    prespike_df = add_zscores(prespike_df, baselines)

    # Statistical tests
    print("Running statistical tests...")
    test_results = run_statistical_tests(prespike_df)

    # Save outputs
    spike_events.to_csv(os.path.join(OUTPUT_DIR, "spike_events.csv"), index=False)
    prespike_df.to_csv(os.path.join(OUTPUT_DIR, "prespike_features.csv"), index=False)
    test_results.to_csv(os.path.join(OUTPUT_DIR, "statistical_tests.csv"), index=False)

    # Visualizations
    print("Generating visualizations...")
    plot_spike_aligned_features(prespike_df, os.path.join(OUTPUT_DIR, "feature_lead_plots.png"))
    plot_wa_vs_or(prespike_df, os.path.join(OUTPUT_DIR, "wa_vs_or_comparison.png"))

    # Summary
    print_summary(spike_events, test_results)


if __name__ == "__main__":
    main()

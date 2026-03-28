#!/usr/bin/env python3
"""
Generate all publication-quality figures for the DATect MDPI Toxins paper.

Usage:
    python3 paper/generate_figures.py

Reads from:
    cache/retrospective/*.parquet     (seed=123 independent test set)
    cache/spectral/all_sites.json     (power spectral density)
    cache/visualizations/*_correlation.json  (feature correlations)
    config.py                          (site coordinates)

Outputs to:
    paper/figures/fig1_study_area.png
    paper/figures/fig2_scatter_best_sites.png
    paper/figures/fig3_architecture.png
    paper/figures/fig4_waterfall_timeseries.png
    paper/figures/fig5_feature_importance.png
    paper/figures/fig6_feature_heatmap.png
    paper/figures/fig7_correlation_heatmap.png
    paper/figures/fig8_spectral_density.png
"""

import sys
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from sklearn.metrics import r2_score

# Add project root to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

warnings.filterwarnings('ignore', category=UserWarning)

# ─── Global Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

OUTDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTDIR, exist_ok=True)

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cache')

# DA risk category colors
CAT_COLORS = {0: '#2ca02c', 1: '#f0c929', 2: '#ff7f0e', 3: '#d62728'}
CAT_NAMES = {0: 'Low (0-5)', 1: 'Moderate (5-20)', 2: 'High (20-40)', 3: 'Extreme (>40)'}

# Site ordering by latitude (north to south)
SITE_ORDER = [
    'Kalaloch', 'Quinault', 'Copalis', 'Twin Harbors', 'Long Beach',
    'Clatsop Beach', 'Cannon Beach', 'Newport', 'Coos Bay', 'Gold Beach'
]

# Feature category colors for importance plots
FEAT_CATEGORIES = {
    'persistence': ('#e67e22', [
        'last_observed_da_raw', 'weeks_since_spike', 'last_da_raw_above_20',
    ]),
    'lag': ('#f39c12', [
        'da_raw_prev_obs_1', 'da_raw_prev_obs_2', 'da_raw_prev_obs_3',
        'da_raw_prev_obs_4', 'da_raw_prev_obs_diff_1_2',
        'da_raw_prev_obs_2_weeks_ago', 'da_raw_prev_obs_3_weeks_ago',
    ]),
    'rolling': ('#d35400', [
        'da_raw_rolling_mean_4wk', 'da_raw_rolling_max_4wk',
        'da_raw_rolling_std_8wk', 'da_raw_rolling_max_8wk',
        'da_raw_rolling_std_12wk', 'da_raw_rolling_max_12wk',
    ]),
    'satellite': ('#3498db', [
        'sst', 'sst_anom', 'flh',
    ]),
    'climate': ('#27ae60', [
        'beuti', 'pdo', 'oni', 'discharge',
    ]),
    'temporal': ('#8e44ad', [
        'sin_day_of_year', 'cos_day_of_year', 'month',
    ]),
    'other': ('#95a5a6', [
        'pn_log',
    ]),
}


def get_feat_color(feat_name):
    """Return color for a feature based on its category."""
    for cat, (color, members) in FEAT_CATEGORIES.items():
        if feat_name in members:
            return color
    return '#95a5a6'


def load_ensemble_parquet():
    path = os.path.join(CACHE_DIR, 'retrospective', 'regression_ensemble.parquet')
    return pd.read_parquet(path)


def load_xgboost_parquet():
    path = os.path.join(CACHE_DIR, 'retrospective', 'regression_xgboost.parquet')
    return pd.read_parquet(path)


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 1: Study Area Map
# ═════════════════════════════════════════════════════════════════════════════

def generate_fig1_study_area():
    """Map of the Pacific Northwest coast with 10 monitoring sites."""
    import ssl
    import certifi
    ssl._create_default_https_context = ssl._create_unverified_context

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig, ax = plt.subplots(
        figsize=(4.5, 7),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # Map extent: PNW coast
    ax.set_extent([-126.5, -122.5, 41.5, 48.5], crs=ccrs.PlateCarree())

    # Add geographic features
    ax.add_feature(cfeature.LAND, facecolor='#f0ebe3', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#d4e6f1')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='#555555')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', color='#888888')
    ax.add_feature(cfeature.STATES, linewidth=0.4, linestyle=':', color='#aaaaaa')
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, color='#85c1e9', alpha=0.7)

    # Plot sites
    wa_sites = ['Kalaloch', 'Quinault', 'Copalis', 'Twin Harbors', 'Long Beach']
    or_sites = ['Clatsop Beach', 'Cannon Beach', 'Newport', 'Coos Bay', 'Gold Beach']

    for site_name, (lat, lon) in config.SITES.items():
        color = '#2874a6' if site_name in wa_sites else '#c0392b'
        marker = 'o' if site_name in wa_sites else 's'
        ax.plot(lon, lat, marker=marker, color=color, markersize=7,
                markeredgecolor='white', markeredgewidth=0.8,
                transform=ccrs.PlateCarree(), zorder=5)

        # Label offset to avoid overlap
        x_off, ha = (0.12, 'left')
        if site_name in ['Newport', 'Coos Bay', 'Gold Beach']:
            x_off, ha = (0.12, 'left')
        ax.text(lon + x_off, lat, site_name, fontsize=7,
                ha=ha, va='center', transform=ccrs.PlateCarree(),
                fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.8))

    # Annotate key oceanographic features
    ax.annotate('Juan de Fuca\nEddy', xy=(-125.5, 48.2),
                fontsize=7, fontstyle='italic', color='#2c3e50', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#d4e6f1',
                          edgecolor='#85c1e9', alpha=0.9))

    ax.annotate('Columbia\nRiver', xy=(-123.8, 46.25),
                fontsize=7, fontstyle='italic', color='#2c3e50', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#d4e6f1',
                          edgecolor='#85c1e9', alpha=0.9))

    ax.annotate('Heceta\nBank', xy=(-125.5, 44.0),
                fontsize=7, fontstyle='italic', color='#2c3e50', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#d4e6f1',
                          edgecolor='#85c1e9', alpha=0.9))

    # State labels
    ax.text(-123.5, 47.3, 'WASHINGTON', fontsize=8, color='#666666',
            ha='center', fontstyle='italic', transform=ccrs.PlateCarree())
    ax.text(-123.5, 43.5, 'OREGON', fontsize=8, color='#666666',
            ha='center', fontstyle='italic', transform=ccrs.PlateCarree())

    # Legend
    wa_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2874a6',
                          markersize=7, label='Washington sites')
    or_patch = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#c0392b',
                          markersize=7, label='Oregon sites')
    ax.legend(handles=[wa_patch, or_patch], loc='lower left', fontsize=7,
              framealpha=0.9, edgecolor='#cccccc')

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 7}
    gl.ylabel_style = {'size': 7}

    outpath = os.path.join(OUTDIR, 'fig1_study_area.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 2: Scatter — Predicted vs Actual DA (Best Sites)
# ═════════════════════════════════════════════════════════════════════════════

def generate_fig2_scatter():
    """Scatter plots of predicted vs actual DA for Copalis and Long Beach."""
    df = load_ensemble_parquet()

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    for ax, site in zip(axes, ['Copalis', 'Long Beach']):
        site_df = df[df['site'] == site].copy()
        actual = site_df['actual_da'].values
        predicted = site_df['predicted_da'].values
        categories = site_df['actual_category'].values
        r2 = r2_score(actual, predicted)

        # Plot by category
        for cat in sorted(site_df['actual_category'].unique()):
            mask = categories == cat
            ax.scatter(actual[mask], predicted[mask],
                       c=CAT_COLORS[cat], s=18, alpha=0.6,
                       edgecolors='white', linewidths=0.3,
                       label=CAT_NAMES[cat], zorder=3)

        # Perfect prediction line
        max_val = max(actual.max(), predicted.max()) * 1.05
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.8, alpha=0.5,
                label='$y = x$', zorder=2)

        # 20 µg/g threshold lines
        ax.axhline(y=20, color='red', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.axvline(x=20, color='red', linewidth=0.5, linestyle=':', alpha=0.5)

        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_xlabel('Actual DA (µg/g)')
        ax.set_ylabel('Predicted DA (µg/g)')
        ax.set_title(f'{site} ($R^2 = {r2:.3f}$, $n = {len(site_df)}$)')
        ax.set_aspect('equal')

    # Single legend for both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.02), frameon=True, edgecolor='#cccccc')

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    outpath = os.path.join(OUTDIR, 'fig2_scatter_best_sites.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 3: System Architecture Diagram
# ═════════════════════════════════════════════════════════════════════════════

def generate_fig3_architecture():
    """System architecture diagram showing data flow through DATect."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.2, 8.2)
    ax.axis('off')

    def draw_box(x, y, w, h, text, color, fontsize=7, bold=False):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='#555555', linewidth=0.8)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, wrap=True)

    def draw_arrow(x1, y1, x2, y2, style='->', color='#555555', lw=1.2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle=style, color=color,
                                     linewidth=lw, connectionstyle='arc3,rad=0'))

    def draw_label(x, y, text, fontsize=8, color='#2c3e50'):
        ax.text(x, y, text, ha='center', fontsize=fontsize,
                fontweight='bold', color=color)

    # ── Layout constants ──
    cx = 5.0  # canvas center

    # ═══ Row 1: Data Sources ═══
    y1, h1 = 6.8, 0.85
    box_w1 = 1.9
    gap1 = 0.2
    total_w1 = 4 * box_w1 + 3 * gap1
    x1_start = cx - total_w1 / 2

    sources = [
        'MODIS-Aqua\nSatellite\n(SST, Chl, FLH)',
        'NOAA ERDDAP\n(BEUTI, PDO,\nONI)',
        'USGS NWIS\n(Columbia R.\ndischarge)',
        'WDOH / ODFW\n(DA conc.,\nPN counts)',
    ]
    src_centers = []
    for i, label in enumerate(sources):
        x = x1_start + i * (box_w1 + gap1)
        draw_box(x, y1, box_w1, h1, label, '#d5e8d4', fontsize=6.5)
        src_centers.append(x + box_w1 / 2)

    draw_label(cx, y1 + h1 + 0.2, 'Data Sources', fontsize=9)

    # ═══ Row 2: Data Pipeline (all sources feed in) ═══
    y2, h2 = 5.4, 0.85
    box_w2 = 5.5
    l2_x = cx - box_w2 / 2
    l2_cx = cx

    draw_box(l2_x, y2, box_w2, h2,
             'Data Pipeline (dataset-creation.py)\nFeature engineering, gap-filling, lag computation',
             '#dae8fc', fontsize=7)

    # Arrows: all 4 data sources → pipeline
    for scx in src_centers:
        draw_arrow(scx, y1, l2_cx + (scx - l2_cx) * 0.4, y2 + h2)

    # ═══ Row 3: Processed Dataset output ═══
    y3, h3 = 4.2, 0.65
    box_w3 = 3.2
    l3_x = cx - box_w3 / 2
    l3_cx = cx

    draw_box(l3_x, y3, box_w3, h3,
             'Processed Dataset\n(final_output.parquet)',
             '#e8e8e8', fontsize=7, bold=False)

    # Arrow: pipeline → parquet
    draw_arrow(l2_cx, y2, l3_cx, y3 + h3)

    # ═══ Row 4: Forecasting Engine (center) + side inputs ═══
    y4, h4 = 2.7, 0.95

    # Per-site config (left side input)
    cfg_w = 2.2
    cfg_x = 0.3
    cfg_cx = cfg_x + cfg_w / 2
    draw_box(cfg_x, y4 + 0.1, cfg_w, h4 - 0.2,
             'Per-Site Config\n(per_site_models.py)\nHyperparams, features,\nensemble weights',
             '#e1d5e7', fontsize=6)

    # Raw DA measurements (right side input)
    raw_w = 2.0
    raw_x = 10 - 0.3 - raw_w
    raw_cx = raw_x + raw_w / 2
    draw_box(raw_x, y4 + 0.1, raw_w, h4 - 0.2,
             'Raw DA\nMeasurements\n(CSV files)',
             '#d5e8d4', fontsize=6)

    # Forecasting engine (center)
    eng_w = 4.6
    eng_x = cx - eng_w / 2
    eng_cx = cx
    draw_box(eng_x, y4, eng_w, h4,
             'Forecasting Engine (raw_forecast_engine.py)\nPer-site XGBoost/RF  |  Leak-free validation\n|  Spike classifier',
             '#fff2cc', fontsize=7, bold=True)

    # Arrow: parquet → engine
    draw_arrow(l3_cx, y3, eng_cx, y4 + h4)

    # Arrow: per-site config → engine (horizontal)
    draw_arrow(cfg_x + cfg_w, y4 + h4 / 2, eng_x, y4 + h4 / 2)

    # Arrow: raw DA → engine (horizontal)
    draw_arrow(raw_x, y4 + h4 / 2, eng_x + eng_w, y4 + h4 / 2)

    # Dashed arrow: WDOH/ODFW data source also feeds raw DA directly
    # (raw DA CSVs are read by both pipeline and engine)
    ax.annotate('', xy=(raw_cx, y4 + h4 - 0.2 + 0.1),
                xytext=(src_centers[3], y1),
                arrowprops=dict(arrowstyle='->', color='#999999',
                                linewidth=0.9, linestyle='dashed',
                                connectionstyle='arc3,rad=-0.15'))

    # ═══ Row 5: Serving layer ═══
    y5, h5 = 1.1, 0.75

    # Cache (left)
    cache_w = 2.5
    cache_x = cx - cache_w - 0.6
    cache_cx = cache_x + cache_w / 2
    draw_box(cache_x, y5, cache_w, h5,
             'Pre-computed Cache\n(+ optional Redis)',
             '#e1d5e7', fontsize=7)

    # Backend API (right)
    api_w = 2.5
    api_x = cx + 0.6
    api_cx = api_x + api_w / 2
    draw_box(api_x, y5, api_w, h5,
             'FastAPI Backend\n(api.py)',
             '#f8cecc', fontsize=7)

    # Arrows: engine → cache, engine → API
    draw_arrow(eng_cx - 0.8, y4, cache_cx, y5 + h5)
    draw_arrow(eng_cx + 0.8, y4, api_cx, y5 + h5)

    # Arrow: cache → API (horizontal)
    draw_arrow(cache_x + cache_w, y5 + h5 / 2, api_x, y5 + h5 / 2)

    # ═══ Row 6: Frontend ═══
    y6, h6 = 0.0, 0.7
    fe_w = 2.5
    fe_x = cx - fe_w / 2
    fe_cx = cx
    draw_box(fe_x, y6, fe_w, h6,
             'React Frontend\n(dashboard)',
             '#f8cecc', fontsize=7)

    # Arrow: API → frontend
    draw_arrow(api_cx, y5, fe_cx + 0.3, y6 + h6)

    outpath = os.path.join(OUTDIR, 'fig3_architecture.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 5: Feature Importance Bar Chart (Top 15)
# ═════════════════════════════════════════════════════════════════════════════

def generate_fig5_feature_importance():
    """Horizontal bar chart of top 15 features by mean XGBoost gain importance."""
    df = load_xgboost_parquet()

    # Parse feature importance dicts and average across all predictions
    all_fi = []
    for fi in df['feature_importance']:
        if isinstance(fi, str):
            fi = json.loads(fi)
        if isinstance(fi, dict):
            all_fi.append(fi)

    # Aggregate: mean importance per feature
    fi_df = pd.DataFrame(all_fi).fillna(0)
    mean_importance = fi_df.mean().sort_values(ascending=False)

    # Top 15
    top15 = mean_importance.head(15)

    fig, ax = plt.subplots(figsize=(5, 4))

    colors = [get_feat_color(f) for f in top15.index]
    bars = ax.barh(range(len(top15)), top15.values, color=colors,
                   edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels([f.replace('_', ' ').replace('da raw ', 'DA ')
                         .replace('prev obs', 'lag')
                         .replace('rolling ', '')
                         for f in top15.index], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Mean XGBoost Gain Importance')
    ax.set_title('Top 15 Features by Importance (averaged across all sites)')

    # Category legend
    cat_handles = []
    seen = set()
    for cat, (color, _) in FEAT_CATEGORIES.items():
        if any(f in top15.index for f in FEAT_CATEGORIES[cat][1]) and cat not in seen:
            cat_handles.append(mpatches.Patch(color=color, label=cat.capitalize()))
            seen.add(cat)
    ax.legend(handles=cat_handles, loc='lower right', fontsize=7,
              framealpha=0.9, edgecolor='#cccccc')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    outpath = os.path.join(OUTDIR, 'fig5_feature_importance.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 6: Feature Importance Heatmap (Features × Sites)
# ═════════════════════════════════════════════════════════════════════════════

def generate_fig6_feature_heatmap():
    """Heatmap of feature importance across sites."""
    df = load_xgboost_parquet()

    # Compute mean importance per site
    site_fi = {}
    for site in SITE_ORDER:
        site_df = df[df['site'] == site]
        fi_list = []
        for fi in site_df['feature_importance']:
            if isinstance(fi, str):
                fi = json.loads(fi)
            if isinstance(fi, dict):
                fi_list.append(fi)
        if fi_list:
            site_fi[site] = pd.DataFrame(fi_list).fillna(0).mean()

    fi_matrix = pd.DataFrame(site_fi)

    # Select top 15 features by overall mean
    top_feats = fi_matrix.mean(axis=1).sort_values(ascending=False).head(15).index
    fi_matrix = fi_matrix.loc[top_feats]

    # Clean feature names for display
    clean_names = {f: f.replace('_', ' ').replace('da raw ', 'DA ')
                       .replace('prev obs', 'lag')
                       .replace('rolling ', '')
                   for f in fi_matrix.index}
    fi_matrix = fi_matrix.rename(index=clean_names)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(fi_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'XGBoost Gain Importance', 'shrink': 0.8},
                annot_kws={'size': 6})

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Feature Importance by Site (XGBoost gain, top 15 features)')

    # Color site names by state
    for i, label in enumerate(ax.get_xticklabels()):
        site_name = label.get_text()
        if site_name in ['Kalaloch', 'Quinault', 'Copalis', 'Twin Harbors', 'Long Beach']:
            label.set_color('#2874a6')
        else:
            label.set_color('#c0392b')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    fig.tight_layout()
    outpath = os.path.join(OUTDIR, 'fig6_feature_heatmap.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 7: Correlation Heatmap (Twin Harbors)
# ═════════════════════════════════════════════════════════════════════════════

def generate_fig7_correlation():
    """Feature correlation heatmap for Twin Harbors, converted from Plotly cache."""
    from matplotlib.colors import LinearSegmentedColormap

    path = os.path.join(CACHE_DIR, 'visualizations', 'Twin Harbors_correlation.json')
    with open(path) as f:
        plotly_data = json.load(f)

    z = np.array(plotly_data['data'][0]['z'])
    x_labels = plotly_data['data'][0]['x']
    y_labels = plotly_data['data'][0]['y']

    # Clean labels
    clean = lambda s: s.replace('_', ' ').replace('da raw ', 'DA ') if isinstance(s, str) else str(s)
    x_clean = [clean(l) for l in x_labels]
    y_clean = [clean(l) for l in y_labels]

    # Custom colormap matching frontend: red (negative) → white (zero) → blue (positive)
    frontend_colors = [
        (0.0,  (178/255, 24/255, 43/255)),   # -1.0: dark red
        (0.25, (239/255, 138/255, 98/255)),   # -0.5: light red
        (0.5,  (1.0, 1.0, 1.0)),             #  0.0: white
        (0.75, (103/255, 169/255, 207/255)),  # +0.5: light blue
        (1.0,  (33/255, 102/255, 172/255)),   # +1.0: dark blue
    ]
    cmap = LinearSegmentedColormap.from_list('frontend_rdbu', frontend_colors)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Full matrix (no triangular mask) — matching frontend display
    # Build annotation array with conditional font colors
    annot_colors = np.where(np.abs(z) > 0.7, 'white', 'black')

    sns.heatmap(z, xticklabels=x_clean, yticklabels=y_clean,
                cmap=cmap, center=0, vmin=-1, vmax=1,
                linewidths=0.3, linecolor='white', ax=ax,
                cbar_kws={'label': 'Pearson Correlation (r)', 'shrink': 0.8,
                          'ticks': [-1, -0.5, 0, 0.5, 1]},
                annot=True, fmt='.2f', annot_kws={'size': 5.5})

    # Apply conditional font colors (white text on dark cells)
    for i, text_row in enumerate(ax.texts):
        row_idx = i // z.shape[1]
        col_idx = i % z.shape[1]
        text_row.set_color(annot_colors[row_idx, col_idx])

    ax.set_title('Feature Correlation Matrix — Twin Harbors')
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    ax.tick_params(axis='y', rotation=0, labelsize=6)

    fig.tight_layout()
    outpath = os.path.join(OUTDIR, 'fig7_correlation_heatmap.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 8: Power Spectral Density
# ═════════════════════════════════════════════════════════════════════════════

def generate_fig8_spectral():
    """Power spectral density plot from cached Plotly data."""
    path = os.path.join(CACHE_DIR, 'spectral', 'all_sites.json')
    with open(path) as f:
        plots = json.load(f)

    # The first plot in the list is typically the PSD
    psd_plot = plots[0]

    fig, ax = plt.subplots(figsize=(6, 4))

    colors = ['#2874a6', '#c0392b', '#27ae60', '#8e44ad', '#e67e22']
    for i, trace in enumerate(psd_plot['data']):
        x = trace.get('x', [])
        y = trace.get('y', [])
        name = trace.get('name', f'Trace {i}')
        color = colors[i % len(colors)]
        mode = trace.get('mode', 'lines')

        if 'lines' in mode:
            ax.plot(x, y, label=name, color=color, linewidth=1.2, alpha=0.8)
        if 'markers' in mode:
            ax.scatter(x, y, label=name, color=color, s=10, alpha=0.8)

    # Check if the plot uses log scale
    layout = psd_plot.get('layout', {})
    xaxis = layout.get('xaxis', {})
    yaxis = layout.get('yaxis', {})

    if xaxis.get('type') == 'log':
        ax.set_xscale('log')
    if yaxis.get('type') == 'log':
        ax.set_yscale('log')

    # Handle both string and dict title formats from Plotly JSON
    def get_title(obj, key, default=''):
        val = obj.get(key, default)
        if isinstance(val, dict):
            return val.get('text', default)
        return val if isinstance(val, str) else default

    ax.set_xlabel(get_title(xaxis, 'title', 'Frequency'))
    ax.set_ylabel(get_title(yaxis, 'title', 'Power Spectral Density'))
    ax.set_title(get_title(layout, 'title', 'Power Spectral Density — All Sites'))

    ax.legend(fontsize=7, framealpha=0.9, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    outpath = os.path.join(OUTDIR, 'fig8_spectral_density.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 4: Waterfall Time Series (DA across all sites)
# ═════════════════════════════════════════════════════════════════════════════

def generate_fig4_waterfall():
    """Waterfall (ridge) plot of DA concentrations across all 10 sites over time."""
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'final_output.parquet')
    data = pd.read_parquet(DATA_PATH)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['lat', 'date'])

    lat_to_site = data.groupby('lat')['site'].first().to_dict()
    unique_lats = sorted(data['lat'].unique(), reverse=True)  # north to south

    BASELINE_MULT = 3
    DA_SCALE = 0.01

    # Use a 10-color qualitative palette matching Plotly defaults
    prop_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, lat in enumerate(unique_lats):
        site_name = lat_to_site.get(lat, f"Lat {lat:.2f}")
        site_data = data[data['lat'] == lat].sort_values('date')

        baseline_y = lat * BASELINE_MULT
        da_vals = site_data['da'].fillna(0).values
        y_vals = baseline_y + DA_SCALE * da_vals
        dates = site_data['date'].values

        color = prop_colors[i % len(prop_colors)]

        # Fill between baseline and DA trace for visual weight
        ax.fill_between(dates, baseline_y, y_vals, alpha=0.25, color=color, linewidth=0)
        ax.plot(dates, y_vals, linewidth=0.8, color=color, label=site_name)

        # Baseline tick line
        ax.axhline(y=baseline_y, color='#cccccc', linewidth=0.3, zorder=0)

        # Reference bars for DA = 20, 50, 100 µg/g
        bar_date = pd.Timestamp('2017-06-01')
        for da_level, ls in [(20, ':'), (50, '--'), (100, '-')]:
            y_ref = baseline_y + DA_SCALE * da_level
            ax.plot([bar_date, bar_date], [baseline_y, y_ref],
                    color='gray', linewidth=1.0, alpha=0.5, linestyle=ls)
            ax.text(bar_date + pd.Timedelta(days=30), y_ref, f'{da_level}',
                    fontsize=4.5, color='gray', va='bottom', ha='left')

    # Y-axis: site labels at baseline positions
    y_ticks = [lat * BASELINE_MULT for lat in unique_lats]
    y_labels = [f"{lat_to_site.get(lat, '')} ({lat:.1f}°N)" for lat in unique_lats]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=6.5)

    ax.set_xlabel('Date')
    ax.set_ylabel('')
    ax.set_title('Domoic Acid Concentrations Across All Sites Over Time\n'
                 '(reference bars: DA = 20, 50, 100 µg/g)', fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(loc='upper left', fontsize=5.5, ncol=2, framealpha=0.9,
              edgecolor='#cccccc', handlelength=1.5)

    fig.tight_layout()
    outpath = os.path.join(OUTDIR, 'fig4_waterfall_timeseries.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("DATect Paper Figure Generation")
    print("=" * 50)
    print(f"Output directory: {OUTDIR}")
    print(f"Cache directory:  {CACHE_DIR}")
    print()

    generators = [
        ("Figure 1: Study Area Map", generate_fig1_study_area),
        ("Figure 2: Scatter (Copalis, Long Beach)", generate_fig2_scatter),
        ("Figure 3: System Architecture", generate_fig3_architecture),
        ("Figure 4: Waterfall Time Series", generate_fig4_waterfall),
        ("Figure 5: Feature Importance (Top 15)", generate_fig5_feature_importance),
        ("Figure 6: Feature Importance Heatmap", generate_fig6_feature_heatmap),
        ("Figure 7: Correlation Heatmap (Twin Harbors)", generate_fig7_correlation),
        ("Figure 8: Power Spectral Density", generate_fig8_spectral),
    ]

    for name, func in generators:
        print(f"Generating {name}...")
        try:
            func()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("Done! All figures saved to paper/figures/")


if __name__ == '__main__':
    main()

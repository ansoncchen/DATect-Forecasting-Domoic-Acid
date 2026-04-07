#!/usr/bin/env python3
"""
Generate all publication-quality figures for the DATect MDPI Toxins paper.

Usage:
    python3 paper/generate_figures.py

Reads from:
    cache/retrospective/*.parquet     (seed=123 held-out test set; used where available)
    cache/spectral/all_sites.json     (power spectral density)
    cache/visualizations/*_correlation.json  (feature correlations)
    config.py                          (site coordinates)

Notes:
    Figure 2 and Figure 4 are manuscript-synced summary panels built from
    paper-reported metrics/constants so they stay aligned with the final tables
    even when the retrospective cache is unavailable locally.

Outputs to:
    paper/figures/fig1_study_area.png
    paper/figures/fig2_site_performance.png
    paper/figures/fig3_architecture.png
    paper/figures/fig3b_ml_pipeline.png
    paper/figures/fig4_spike_detection_summary.png
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
        'last_observed_da_raw', 'weeks_since_last_spike',
    ]),
    'lag': ('#f39c12', [
        'da_raw_prev_obs_1', 'da_raw_prev_obs_2', 'da_raw_prev_obs_3',
        'da_raw_prev_obs_4', 'da_raw_prev_obs_diff_1_2',
        'da_raw_prev_obs_2_weeks_ago', 'da_raw_prev_obs_3_weeks_ago',
        'da_raw_prev_obs_4_weeks_ago',
    ]),
    'rolling': ('#d35400', [
        'raw_obs_roll_mean_4', 'raw_obs_roll_std_4', 'raw_obs_roll_max_4',
        'raw_obs_roll_std_8', 'raw_obs_roll_max_8',
        'raw_obs_roll_std_12', 'raw_obs_roll_max_12',
    ]),
    'satellite': ('#3498db', [
        'modis-sst', 'sst-anom', 'flh',
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
#  FIGURE 2: Per-Site Held-Out Performance
# ═════════════════════════════════════════════════════════════════════════════

def generate_fig2_scatter():
    """Per-site held-out R^2 plot supporting the WA/OR performance split."""
    sites = [
        'Copalis', 'Long Beach', 'Twin Harbors', 'Quinault', 'Kalaloch',
        'Clatsop Beach', 'Gold Beach', 'Cannon Beach', 'Coos Bay', 'Newport',
    ]
    r2_values = np.array([0.789, 0.631, 0.594, 0.582, 0.480, 0.290, 0.041, -0.044, -0.039, -0.299])
    n_values = [277, 209, 225, 181, 235, 354, 257, 116, 109, 218]
    colors = ['#2874a6' if site in SITE_ORDER[:5] else '#c0392b' for site in sites]

    y_pos = np.arange(len(sites))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axvline(0, color='#777777', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.hlines(y_pos, 0, r2_values, color=colors, linewidth=2.2, alpha=0.85)
    ax.scatter(r2_values, y_pos, s=50, color=colors, edgecolor='white', linewidth=0.8, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sites, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r'Held-out $R^2$')
    ax.set_title('Per-site held-out regression performance')
    ax.set_xlim(-0.4, 0.85)
    ax.xaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for x, y, n, color in zip(r2_values, y_pos, n_values, colors):
        offset = 0.03 if x >= 0 else -0.03
        ha = 'left' if x >= 0 else 'right'
        ax.text(x + offset, y, f'n={n}', va='center', ha=ha, fontsize=7, color=color)

    wa_patch = mpatches.Patch(color='#2874a6', label='Washington')
    or_patch = mpatches.Patch(color='#c0392b', label='Oregon')
    ax.legend(handles=[wa_patch, or_patch], loc='lower right', fontsize=7,
              framealpha=0.9, edgecolor='#cccccc')

    fig.tight_layout()
    outpath = os.path.join(OUTDIR, 'fig2_site_performance.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 3: System Architecture Diagram
# ═════════════════════════════════════════════════════════════════════════════

def generate_fig3_architecture():
    """System architecture diagram showing data flow through DATect (Graphviz)."""
    import graphviz

    dot = graphviz.Digraph('DATect', format='png')
    dot.attr(rankdir='TB', dpi='300', bgcolor='white',
             fontname='Helvetica', fontsize='10',
             pad='0.4', nodesep='0.4', ranksep='0.45')

    # Default node style
    dot.attr('node', shape='box', style='filled,rounded',
             fontname='Helvetica', fontsize='9', penwidth='1.0',
             color='#666666', margin='0.15,0.1')
    dot.attr('edge', color='#666666', arrowsize='0.7', penwidth='1.0')

    # ── Colors ──
    C_SRC  = '#d5e8d4'
    C_PROC = '#dae8fc'
    C_DATA = '#f0f0f0'
    C_ML   = '#fff2cc'
    C_SERVE = '#f8cecc'

    # ═══ Row 1: Data Sources (ordered left→right) ═══
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('modis', 'MODIS-Aqua Satellite\n(SST, Chl, FLH)', fillcolor=C_SRC)
        s.node('erddap', 'NOAA ERDDAP\n(BEUTI, PDO, ONI)', fillcolor=C_SRC)
        s.node('usgs', 'USGS NWIS\n(Columbia R. discharge)', fillcolor=C_SRC)
        s.node('wdoh', 'WDOH / ODFW\n(DA conc., PN counts)', fillcolor=C_SRC)
        # Invisible edges to enforce left-to-right ordering
        s.edge('modis', 'erddap', style='invis')
        s.edge('erddap', 'usgs', style='invis')
        s.edge('usgs', 'wdoh', style='invis')

    # ═══ Row 2: Data Pipeline ═══
    dot.node('pipeline', 'Data Pipeline  (dataset-creation.py)\n'
             'Feature engineering, gap-filling, lag computation',
             fillcolor=C_PROC, fontsize='10')

    # ═══ Row 3: Data Artifacts (ordered left→right) ═══
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('parquet', 'Processed Dataset\n(final_output.parquet)',
               fillcolor=C_DATA, style='filled,rounded,dashed')
        s.node('rawda', 'Raw DA Measurements\n(CSV files)',
               fillcolor=C_DATA, style='filled,rounded,dashed')
        s.edge('parquet', 'rawda', style='invis')

    # ═══ Row 4: Forecasting Engine ═══
    dot.node('engine',
             'Forecasting Engine  (raw_forecast_engine.py)\n'
             'Per-site XGBoost/RF ensemble  |  Leak-free validation  |  Spike classifier\n'
             'Per-site config: hyperparameters, feature subsets, ensemble weights  (per_site_models.py)',
             fillcolor=C_ML, fontsize='9', penwidth='1.5')

    # ═══ Row 5: Serving (ordered left→right) ═══
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('cache', 'Pre-computed Cache\n(+ optional Redis)', fillcolor=C_SERVE)
        s.node('api', 'FastAPI Backend\n(api.py)', fillcolor=C_SERVE)
        s.node('frontend', 'React Frontend\n(dashboard)', fillcolor=C_SERVE)

    # ── Edges ──
    # Data sources → pipeline
    dot.edge('modis', 'pipeline')
    dot.edge('erddap', 'pipeline')
    dot.edge('usgs', 'pipeline')
    dot.edge('wdoh', 'pipeline')

    # Pipeline → processed dataset
    dot.edge('pipeline', 'parquet')

    # WDOH/ODFW also provides raw DA directly (bypasses pipeline)
    dot.edge('wdoh', 'rawda', style='dashed', color='#999999')

    # Data artifacts → engine
    dot.edge('parquet', 'engine')
    dot.edge('rawda', 'engine')

    # Engine → serving
    dot.edge('engine', 'cache')
    dot.edge('engine', 'api')

    # Cache → API → Frontend
    dot.edge('cache', 'api')
    dot.edge('api', 'frontend')

    outpath = os.path.join(OUTDIR, 'fig3_architecture')
    dot.render(outpath, cleanup=True)
    print(f"  Saved: {outpath}.png")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 3b: ML Pipeline Detail (inside the Forecasting Engine)
# ═════════════════════════════════════════════════════════════════════════════

def generate_fig3b_ml_pipeline():
    """Detailed ML pipeline diagram showing internals of the Forecasting Engine."""
    import graphviz

    dot = graphviz.Digraph('MLPipeline', format='png')
    dot.attr(rankdir='TB', dpi='300', bgcolor='white',
             fontname='Helvetica', fontsize='10',
             pad='0.4', nodesep='0.35', ranksep='0.4',
             compound='true')

    dot.attr('node', shape='box', style='filled,rounded',
             fontname='Helvetica', fontsize='9', penwidth='1.0',
             color='#666666', margin='0.15,0.1')
    dot.attr('edge', color='#666666', arrowsize='0.7', penwidth='1.0')

    # ── Colors ──
    C_INPUT  = '#d5e8d4'   # green  — inputs
    C_FEAT   = '#dae8fc'   # blue   — feature construction
    C_MODEL  = '#fff2cc'   # yellow — model training/prediction
    C_BLEND  = '#ffe0b2'   # orange — ensemble blending
    C_CLASS  = '#e1d5e7'   # purple — classification
    C_OUTPUT = '#f8cecc'   # pink   — outputs
    C_DATA   = '#f0f0f0'   # gray   — data artifacts

    # ═══ Row 1: Inputs to the engine ═══
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('parquet', 'Processed Dataset\n(environmental features,\ngap-filled DA)',
               fillcolor=C_INPUT)
        s.node('rawda', 'Raw DA Observations\n(real measurements only)',
               fillcolor=C_INPUT)
        s.node('siteconfig', 'Per-Site Config\n(hyperparams, feature subsets,\nensemble weights)',
               fillcolor=C_INPUT)
        s.edge('parquet', 'rawda', style='invis')
        s.edge('rawda', 'siteconfig', style='invis')

    # ═══ Row 2: Expanding window + feature construction ═══
    dot.node('window',
             'Expanding Window Split\nTrain: all data ≤ anchor date  |  '
             'Test: anchor date snapshot\n'
             'Fresh model trained per forecast point',
             fillcolor=C_FEAT, fontsize='9')

    dot.node('features',
             'Feature Construction\n'
             'Observation-order lags (4 values + recency + trend)\n'
             'Rolling stats (mean/max/std at 4/8/12-week)\n'
             'Persistence (last DA, weeks since spike)\n'
             'Environmental (SST, BEUTI, PDO, ONI, discharge, FLH)\n'
             'Temporal (sin/cos day-of-year, month)',
             fillcolor=C_FEAT, fontsize='8')

    # ═══ Row 3: Parallel model training ═══
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('xgb', 'XGBoost Regressor\n(per-site hyperparams)',
               fillcolor=C_MODEL)
        s.node('rf', 'Random Forest Regressor\n(per-site hyperparams)',
               fillcolor=C_MODEL)
        s.node('naive', 'Naïve Persistence\n(last observed DA)',
               fillcolor=C_DATA, style='filled,rounded,dashed')
        s.edge('xgb', 'rf', style='invis')
        s.edge('rf', 'naive', style='invis')

    # ═══ Row 4: Ensemble + classification (parallel) ═══
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('ensemble',
               'Per-Site Weighted Ensemble\n'
               'ŷ = w_xgb · XGB + w_rf · RF\n'
               '(weights tuned per site on dev set)',
               fillcolor=C_BLEND, fontsize='8')
        s.node('classifier',
               'Spike Classifier\n'
               'Dedicated XGBoost classifier\n'
               '4 DA risk categories',
               fillcolor=C_CLASS, fontsize='8')
        s.edge('ensemble', 'classifier', style='invis')

    # ═══ Row 5: Post-processing ═══
    dot.node('postproc',
             'Post-processing\n'
             'Prediction clipping (per-site quantile bounds)\n'
             'Confidence intervals (quantile regression + bootstrap)\n'
             'Threshold classification: Low / Moderate / High / Extreme',
             fillcolor=C_BLEND, fontsize='8')

    # ═══ Row 6: Outputs ═══
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('reg_out', 'Regression Output\nPredicted DA (µg/g)\n+ 90% confidence interval',
               fillcolor=C_OUTPUT)
        s.node('class_out', 'Classification Output\nDA risk category\n+ class probabilities',
               fillcolor=C_OUTPUT)
        s.node('spike_out', 'Spike Detection\nSpike probability\n+ alert flag',
               fillcolor=C_OUTPUT)
        s.edge('reg_out', 'class_out', style='invis')
        s.edge('class_out', 'spike_out', style='invis')

    # ── Edges ──
    # Inputs → window
    dot.edge('parquet', 'window')
    dot.edge('rawda', 'window')
    dot.edge('siteconfig', 'window', style='dashed', color='#999999')

    # Window → features
    dot.edge('window', 'features')

    # Features → parallel models
    dot.edge('features', 'xgb')
    dot.edge('features', 'rf')
    dot.edge('features', 'naive')

    # Models → ensemble
    dot.edge('xgb', 'ensemble')
    dot.edge('rf', 'ensemble')

    # Features → classifier (trained on same data)
    dot.edge('features', 'classifier')

    # Naive → shown as external baseline (dashed to ensemble)
    dot.edge('naive', 'ensemble', style='dashed', color='#999999')

    # Ensemble + classifier → post-processing
    dot.edge('ensemble', 'postproc')
    dot.edge('classifier', 'postproc')

    # Post-processing → outputs
    dot.edge('postproc', 'reg_out')
    dot.edge('postproc', 'class_out')
    dot.edge('postproc', 'spike_out')

    outpath = os.path.join(OUTDIR, 'fig3b_ml_pipeline')
    dot.render(outpath, cleanup=True)
    print(f"  Saved: {outpath}.png")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 4: Spike Detection Summary
# ═════════════════════════════════════════════════════════════════════════════

def generate_fig4_spike_summary():
    """Bar-chart summary of held-out spike-detection performance."""
    approaches = ["Naive", "Regression\n(>=20)", "Spike\nclassifier"]
    transition_recall = np.array([0.236, 0.124, 0.652])
    spike_recall = np.array([0.754, 0.558, 0.815])
    precision = np.array([0.546, 0.618, 0.339])
    colors = ['#7f8c8d', '#2874a6', '#c0392b']

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2), sharey=False)
    metrics = [
        ("Transition recall", transition_recall, "Below-to-above threshold"),
        ("Spike recall", spike_recall, "All DA >= 20 ug/g events"),
        ("Precision", precision, "Fraction of alerts that were spikes"),
    ]

    for ax, (title, values, subtitle) in zip(axes, metrics):
        bars = ax.bar(approaches, values, color=colors, edgecolor='white', linewidth=0.7)
        ax.set_ylim(0, 0.9)
        ax.set_title(title, fontsize=10)
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=7, color='#555555')
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                    f"{value:.3f}", ha='center', va='bottom', fontsize=7)

    fig.suptitle(
        'Held-out spike-detection summary (seed=123, n=2,181, transitions=89)',
        fontsize=10,
        y=1.03,
    )
    fig.tight_layout()
    outpath = os.path.join(OUTDIR, 'fig4_spike_detection_summary.png')
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
        ("Figure 2: Per-Site Performance", generate_fig2_scatter),
        ("Figure 3a: System Architecture", generate_fig3_architecture),
        ("Figure 3b: ML Pipeline Detail", generate_fig3b_ml_pipeline),
        ("Figure 4: Spike Detection Summary", generate_fig4_spike_summary),
        ("Appendix Figure: Waterfall Time Series", generate_fig4_waterfall),
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

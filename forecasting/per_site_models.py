"""
Per-site model configurations for DATect raw-data DA forecasting.

Each site can override:
  - xgb_params: XGBoost hyperparameter overrides (merged onto base_params)
  - rf_params: Random Forest hyperparameter overrides (merged onto RF base_params)
  - param_grid: Custom PARAM_GRID for per-anchor XGB tuning (replaces global grid)
  - feature_subset: Explicit list of features to keep (None = use all default features)
  - ensemble_weights: (xgb_weight, rf_weight, naive_weight) tuple (None = use global)
    naive_weight is always 0.0; naive persistence is reported as an external baseline only
  - prediction_clip_q: Custom quantile for prediction clipping (None = use global)
  - prediction_clip_max: Hard ceiling on predictions in ug/g (None = no hard ceiling)
"""

from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# Feature group definitions (for readable subset selection)
# --------------------------------------------------------------------------

PERSISTENCE_FEATURES = [
    'last_observed_da_raw',
    'weeks_since_last_spike',
]

# Observation-order lag features (replacing grid-shift lags)
LAG_FEATURES_SHORT = [
    'da_raw_prev_obs_1',
    'da_raw_prev_obs_2',
    'da_raw_prev_obs_diff_1_2',
]

LAG_FEATURES_FULL = [
    'da_raw_prev_obs_1',
    'da_raw_prev_obs_2',
    'da_raw_prev_obs_3',
    'da_raw_prev_obs_4',
    'da_raw_prev_obs_2_weeks_ago',
    'da_raw_prev_obs_3_weeks_ago',
    'da_raw_prev_obs_diff_1_2',
]

ROLLING_FEATURES_SHORT = [
    'raw_obs_roll_mean_4',
    'raw_obs_roll_max_4',
]

ROLLING_FEATURES_FULL = [
    'raw_obs_roll_mean_4',
    'raw_obs_roll_std_4',
    'raw_obs_roll_max_4',
    'raw_obs_roll_std_8',
    'raw_obs_roll_max_8',
    'raw_obs_roll_std_12',
    'raw_obs_roll_max_12',
]

ENV_FEATURES_CORE = [
    'modis-sst',
    'pdo',
    'beuti',
]

TEMPORAL_FEATURES_CORE = [
    'sin_day_of_year',
    'cos_day_of_year',
    'month',
]

# Conservative RF params for sites where RF R2 < 0.1
RF_CONSERVATIVE = {
    'n_estimators': 200,
    'max_depth': 6,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 0.5,
}

# --------------------------------------------------------------------------
# Climate & anomaly feature groups
# --------------------------------------------------------------------------

CLIMATE_FEATURES_CORE = [
    'oni',
    'sst-anom',
]

# Columbia River discharge (global gauge, negative proxy for DA flushing)
# Strongest signal at northern sites (lat > 46.5°N)
DISCHARGE_FEATURES = [
    'discharge',
]

# Pseudo-nitzschia log transform (compresses heavy-tailed distribution)
PN_FEATURES = [
    'pn_log',
]


# --------------------------------------------------------------------------
# Site-specific configuration dictionary
# --------------------------------------------------------------------------

SITE_SPECIFIC_CONFIGS: Dict[str, Dict[str, Any]] = {

    # ==================================================================
    # PERSISTENCE-DOMINANT SITES
    # ==================================================================

    'Copalis': {
        # Interp-trained: N=167, XGB=+0.763, RF=+0.771, Naive=+0.770, Ens=+0.802.
        'xgb_params': {
            'max_depth': 2, 'n_estimators': 100, 'learning_rate': 0.03,
            'min_child_weight': 10, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
            'gamma': 1.0, 'subsample': 0.7, 'colsample_bytree': 0.7,
        },
        'rf_params': None,
        'param_grid': [
            {'max_depth': 2, 'n_estimators': 100, 'learning_rate': 0.03,
             'min_child_weight': 10},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES + LAG_FEATURES_SHORT
            + ROLLING_FEATURES_SHORT + TEMPORAL_FEATURES_CORE
            + CLIMATE_FEATURES_CORE + DISCHARGE_FEATURES
        ),
        'ensemble_weights': (0.00, 1.00, 0.00),  # RF-only: RF=+0.771 > XGB=+0.763 (blending tested, no improvement)
        'prediction_clip_q': 0.97,
        'prediction_clip_max': None,
    },

    'Kalaloch': {
        # Interp-trained: N=131, XGB=+0.433, RF=+0.502, Naive=+0.631, Ens=+0.681.
        'xgb_params': {
            'max_depth': 2, 'n_estimators': 80, 'learning_rate': 0.02,
            'min_child_weight': 12, 'reg_alpha': 2.0, 'reg_lambda': 10.0,
            'gamma': 2.0, 'subsample': 0.6, 'colsample_bytree': 0.6,
        },
        'rf_params': None,
        'param_grid': [
            {'max_depth': 2, 'n_estimators': 80, 'learning_rate': 0.02,
             'min_child_weight': 12},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES + LAG_FEATURES_SHORT + TEMPORAL_FEATURES_CORE
            + CLIMATE_FEATURES_CORE + DISCHARGE_FEATURES + PN_FEATURES
        ),
        'ensemble_weights': (0.00, 1.00, 0.00),  # RF-only: RF=+0.502 >> XGB=+0.433
        'prediction_clip_q': 0.95,
        'prediction_clip_max': 80.0,
    },

    'Twin Harbors': {
        # Interp-trained: N=138, XGB=+0.601, RF=+0.614, Naive=+0.695, Ens=+0.776.
        'xgb_params': {
            'max_depth': 3, 'n_estimators': 150, 'learning_rate': 0.03,
            'min_child_weight': 8, 'reg_alpha': 0.5, 'reg_lambda': 3.0,
            'gamma': 0.5, 'subsample': 0.8, 'colsample_bytree': 0.8,
        },
        'rf_params': None,
        'param_grid': [
            {'max_depth': 3, 'n_estimators': 150, 'learning_rate': 0.03,
             'min_child_weight': 8},
            {'max_depth': 2, 'n_estimators': 100, 'learning_rate': 0.03,
             'min_child_weight': 10},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES + LAG_FEATURES_FULL
            + ROLLING_FEATURES_SHORT + ['modis-sst', 'pdo']
            + TEMPORAL_FEATURES_CORE
            + CLIMATE_FEATURES_CORE + DISCHARGE_FEATURES
        ),
        'ensemble_weights': (0.00, 1.00, 0.00),  # RF-only: RF=+0.614 > XGB=+0.601
        'prediction_clip_q': 0.98,
        'prediction_clip_max': None,
    },

    'Quinault': {
        # Interp-trained: N=113, XGB=+0.764, RF=+0.771, Naive=+0.702, Ens=+0.854.
        'xgb_params': {
            'max_depth': 3, 'n_estimators': 200, 'learning_rate': 0.03,
            'min_child_weight': 7, 'reg_alpha': 0.3, 'reg_lambda': 2.0,
            'gamma': 0.3, 'subsample': 0.8, 'colsample_bytree': 0.8,
        },
        'rf_params': None,
        'param_grid': [
            {'max_depth': 3, 'n_estimators': 200, 'learning_rate': 0.03,
             'min_child_weight': 7},
            {'max_depth': 2, 'n_estimators': 150, 'learning_rate': 0.05,
             'min_child_weight': 10},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES + LAG_FEATURES_FULL
            + ROLLING_FEATURES_SHORT + ENV_FEATURES_CORE
            + TEMPORAL_FEATURES_CORE
            + CLIMATE_FEATURES_CORE + DISCHARGE_FEATURES
        ),
        'ensemble_weights': (0.00, 1.00, 0.00),  # RF-only: RF=+0.771 > XGB=+0.764
        'prediction_clip_q': 0.98,
        'prediction_clip_max': None,
    },

    # ==================================================================
    # ML-LEANING SITES
    # ==================================================================

    'Long Beach': {
        # Interp-trained: N=140, XGB=+0.569, RF=+0.555, Naive=+0.482, Ens=+0.569.
        'xgb_params': {
            'max_depth': 3, 'n_estimators': 250, 'learning_rate': 0.03,
            'min_child_weight': 7, 'reg_alpha': 0.3, 'reg_lambda': 2.0,
            'gamma': 0.3, 'subsample': 0.8, 'colsample_bytree': 0.8,
        },
        'rf_params': None,
        'param_grid': [
            {'max_depth': 3, 'n_estimators': 250, 'learning_rate': 0.03,
             'min_child_weight': 7},
            {'max_depth': 4, 'n_estimators': 200, 'learning_rate': 0.05,
             'min_child_weight': 5},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES + LAG_FEATURES_FULL
            + ROLLING_FEATURES_FULL + ENV_FEATURES_CORE
            + TEMPORAL_FEATURES_CORE
            + CLIMATE_FEATURES_CORE + DISCHARGE_FEATURES
        ),
        'ensemble_weights': (1.00, 0.00, 0.00),  # XGB-only: XGB=+0.569 > RF=+0.555
        'prediction_clip_q': 0.98,
        'prediction_clip_max': None,
    },

    'Clatsop Beach': {
        # Interp-trained: N=218, XGB=+0.476, RF=+0.398, Naive=+0.007, Ens=+0.481.
        'xgb_params': None,
        'rf_params': None,
        'param_grid': None,
        'feature_subset': None,
        'ensemble_weights': (1.00, 0.00, 0.00),  # XGB-only: XGB=+0.476 >> RF=+0.398
        'prediction_clip_q': None,
        'prediction_clip_max': None,
    },

    'Coos Bay': {
        # Interp-trained: N=67, XGB=+0.310, RF=+0.337, Naive=-0.286, Ens=+0.337.
        # Autocorrelation ceiling: rho=0.054, R²_ceil=0.003 — site is near-unpredictable.
        # Feature set trimmed to 5 (from 20) to reduce overfitting on tiny N.
        # Single param_grid entry — tuning skipped below MIN_TRAINING_FOR_TUNING anyway.
        'xgb_params': {
            'max_depth': 2, 'n_estimators': 80, 'learning_rate': 0.02,
            'min_child_weight': 15, 'reg_alpha': 2.0, 'reg_lambda': 10.0,
            'gamma': 2.0, 'subsample': 0.6, 'colsample_bytree': 0.6,
        },
        'rf_params': dict(RF_CONSERVATIVE),
        'param_grid': [
            {'max_depth': 2, 'n_estimators': 80, 'learning_rate': 0.02,
             'min_child_weight': 15},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES + ['da_raw_prev_obs_1']
            + ['month', 'modis-sst', 'pdo']
        ),
        'ensemble_weights': (0.00, 1.00, 0.00),  # RF-only: RF=+0.337 > XGB=+0.310
        'prediction_clip_q': 0.97,
        'prediction_clip_max': None,
    },

    # ==================================================================
    # STRUGGLE SITES
    # ==================================================================

    'Cannon Beach': {
        # Interp-trained: N=61, XGB=-0.006, RF=-0.006, Naive=-0.167, Ens=-0.001.
        # Autocorrelation ceiling: rho=0.44, R²_ceil=0.197 — limited but real signal.
        # Feature set trimmed to 7 (from 12) to reduce p/N ratio for N=61.
        'xgb_params': {
            'max_depth': 2, 'n_estimators': 80, 'learning_rate': 0.02,
            'min_child_weight': 12, 'reg_alpha': 1.5, 'reg_lambda': 7.0,
            'gamma': 1.5, 'subsample': 0.6, 'colsample_bytree': 0.6,
        },
        'rf_params': dict(RF_CONSERVATIVE),
        'param_grid': [
            {'max_depth': 2, 'n_estimators': 80, 'learning_rate': 0.02,
             'min_child_weight': 12},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES + ['da_raw_prev_obs_1', 'da_raw_prev_obs_2']
            + ['month', 'modis-sst', 'pdo']
        ),
        'ensemble_weights': (0.00, 1.00, 0.00),  # RF-only: both near-zero skill, N=61
        'prediction_clip_q': 0.95,
        'prediction_clip_max': 80.0,
    },

    'Gold Beach': {
        # Interp-trained: N=144, XGB=+0.156, RF=+0.140, Naive=-0.858, Ens=+0.156.
        # Autocorrelation ceiling: rho=0.038, R²_ceil=0.001 — near-unpredictable.
        # Current R²=0.041 already beats the ceiling (env drivers help despite low autocorr).
        # Feature set trimmed to 5 (from 15) to preserve this signal without overfitting.
        'xgb_params': {
            'max_depth': 2, 'n_estimators': 100, 'learning_rate': 0.02,
            'min_child_weight': 15, 'reg_alpha': 2.0, 'reg_lambda': 10.0,
            'gamma': 2.0, 'subsample': 0.6, 'colsample_bytree': 0.6,
        },
        'rf_params': dict(RF_CONSERVATIVE),
        'param_grid': [
            {'max_depth': 2, 'n_estimators': 100, 'learning_rate': 0.02,
             'min_child_weight': 15},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES + ['da_raw_prev_obs_1']
            + ['month', 'modis-sst', 'pdo']
        ),
        'ensemble_weights': (1.00, 0.00, 0.00),  # XGB-only: XGB=+0.156 > RF=+0.140
        'prediction_clip_q': 0.95,
        'prediction_clip_max': None,
    },

    'Newport': {
        # Interp-trained: N=142, XGB=-0.409, RF=-0.550, Naive=-1.572, Ens=-0.382.
        # Autocorrelation ceiling: rho=0.19, R²_ceil=0.036 — site is near-unpredictable.
        # Feature set trimmed to 5 (from 21) to stop active harm from overfitting.
        # Single param_grid entry — no tuning benefit at this N with near-zero signal.
        'xgb_params': {
            'max_depth': 2, 'n_estimators': 100, 'learning_rate': 0.02,
            'min_child_weight': 15, 'reg_alpha': 2.0, 'reg_lambda': 10.0,
            'gamma': 2.0, 'subsample': 0.6, 'colsample_bytree': 0.6,
        },
        'rf_params': dict(RF_CONSERVATIVE),
        'param_grid': [
            {'max_depth': 2, 'n_estimators': 100, 'learning_rate': 0.02,
             'min_child_weight': 15},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES + ['da_raw_prev_obs_1']
            + ['month', 'modis-sst', 'pdo']
        ),
        'ensemble_weights': (1.00, 0.00, 0.00),  # XGB-only: XGB=-0.409 > RF=-0.550
        'prediction_clip_q': 0.98,
        'prediction_clip_max': None,
    },
}


# --------------------------------------------------------------------------
# Default config for sites not in SITE_SPECIFIC_CONFIGS
# --------------------------------------------------------------------------

DEFAULT_SITE_CONFIG: Dict[str, Any] = {
    'xgb_params': None,
    'rf_params': None,
    'param_grid': None,
    'feature_subset': None,
    'ensemble_weights': None,
    'prediction_clip_q': None,
    'prediction_clip_max': None,
}


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def get_site_config(site: str) -> Dict[str, Any]:
    """Return per-site configuration, falling back to defaults."""
    cfg = dict(DEFAULT_SITE_CONFIG)
    cfg.update(SITE_SPECIFIC_CONFIGS.get(site, {}))
    return cfg


def apply_site_xgb_params(base_params: dict, site: str) -> dict:
    """Merge site-specific XGB params onto global base_params."""
    site_cfg = get_site_config(site)
    if site_cfg['xgb_params'] is None:
        return dict(base_params)
    return {**base_params, **site_cfg['xgb_params']}


def apply_site_rf_params(base_params: dict, site: str) -> dict:
    """Merge site-specific RF params onto global base_params."""
    site_cfg = get_site_config(site)
    if site_cfg['rf_params'] is None:
        return dict(base_params)
    return {**base_params, **site_cfg['rf_params']}


def get_site_param_grid(site: str) -> Optional[List[dict]]:
    """Return site-specific PARAM_GRID, or None to use global grid."""
    return get_site_config(site)['param_grid']


def get_site_ensemble_weights(site: str) -> Tuple[float, float, float]:
    """Return (xgb_weight, rf_weight, naive_weight) for this site.

    Default: (0.56, 0.44, 0.00).  naive_weight is always 0.0; naive
    persistence is evaluated as an external standalone baseline only.
    """
    weights = get_site_config(site)['ensemble_weights']
    if weights is not None:
        return weights
    return (0.56, 0.44, 0.00)


def get_site_clip_params(site: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (clip_quantile, clip_max) for prediction clipping."""
    cfg = get_site_config(site)
    return cfg['prediction_clip_q'], cfg['prediction_clip_max']


def compute_site_drop_cols(
    base_drop_cols: list,
    all_columns: list,
    site: str,
) -> list:
    """Extend drop_cols to enforce site-specific feature subset.

    If the site has a feature_subset, any column NOT in the subset
    (and not already in base_drop_cols) gets added to drop_cols.
    """
    cfg = get_site_config(site)
    subset = cfg['feature_subset']
    if subset is None:
        return list(base_drop_cols)

    drops = set(base_drop_cols)
    subset_set = set(subset)
    for col in all_columns:
        if col not in subset_set and col not in drops:
            drops.add(col)
    return list(drops)

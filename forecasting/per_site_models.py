"""
Per-site model configurations for DATect raw-data DA forecasting.

Each site can override:
  - xgb_params: XGBoost hyperparameter overrides (merged onto base_params)
  - rf_params: Random Forest hyperparameter overrides (merged onto RF base_params)
  - param_grid: Custom PARAM_GRID for per-anchor XGB tuning (replaces global grid)
  - feature_subset: Explicit list of features to keep (None = use all default features)
  - ensemble_weights: (xgb_weight, rf_weight, naive_weight) tuple (None = use global)
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
    'raw_obs_roll_mean_8',
    'raw_obs_roll_std_8',
    'raw_obs_roll_max_8',
    'raw_obs_roll_mean_12',
    'raw_obs_roll_std_12',
    'raw_obs_roll_max_12',
]

ENV_FEATURES_CORE = [
    'modis-sst',
    'pdo',
    'modis-chla',
    'beuti',
]

TEMPORAL_FEATURES_CORE = [
    'sin_day_of_year',
    'cos_day_of_year',
    'month',
]

TEMPORAL_FEATURES_FULL = [
    'sin_day_of_year',
    'cos_day_of_year',
    'month',
    'sin_month',
    'cos_month',
    'sin_week_of_year',
    'cos_week_of_year',
    'days_since_start',
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
# Site-specific configuration dictionary
# --------------------------------------------------------------------------

SITE_SPECIFIC_CONFIGS: Dict[str, Dict[str, Any]] = {

    # ==================================================================
    # PERSISTENCE-DOMINANT SITES
    # ==================================================================

    'Copalis': {
        # Leak-free: N=167, XGB=+0.748, RF=+0.765, Naive=+0.715, Ens=+0.761.
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
        ),
        'ensemble_weights': (0.25, 0.45, 0.30),
        'prediction_clip_q': 0.97,
        'prediction_clip_max': None,
    },

    'Kalaloch': {
        # Leak-free: N=131, XGB=+0.677, RF=+0.683, Naive=+0.669, Ens=+0.685.
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
        ),
        'ensemble_weights': (0.20, 0.40, 0.40),
        'prediction_clip_q': 0.95,
        'prediction_clip_max': 80.0,
    },

    'Twin Harbors': {
        # Leak-free: N=138, XGB=+0.582, RF=+0.589, Naive=+0.763, Ens=+0.781.
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
        ),
        'ensemble_weights': (0.10, 0.25, 0.65),
        'prediction_clip_q': 0.98,
        'prediction_clip_max': None,
    },

    'Quinault': {
        # Leak-free: N=113, XGB=+0.604, RF=+0.584, Naive=+0.590, Ens=+0.653.
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
        ),
        'ensemble_weights': (0.35, 0.30, 0.35),
        'prediction_clip_q': 0.98,
        'prediction_clip_max': None,
    },

    # ==================================================================
    # ML-LEANING SITES
    # ==================================================================

    'Long Beach': {
        # Leak-free: N=140, XGB=+0.614, RF=+0.603, Naive=+0.470, Ens=+0.608.
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
        ),
        'ensemble_weights': (0.45, 0.40, 0.15),
        'prediction_clip_q': 0.98,
        'prediction_clip_max': None,
    },

    'Clatsop Beach': {
        # Leak-free: N=218, XGB=+0.264, RF=+0.246, Naive=-0.015, Ens=+0.255.
        'xgb_params': None,
        'rf_params': None,
        'param_grid': None,
        'feature_subset': None,
        'ensemble_weights': (0.50, 0.45, 0.05),
        'prediction_clip_q': None,
        'prediction_clip_max': None,
    },

    'Coos Bay': {
        # Leak-free: N=67, XGB=-0.030, RF=+0.290, Naive=-0.570, Ens=+0.101.
        'xgb_params': {
            'max_depth': 3, 'n_estimators': 200, 'learning_rate': 0.03,
            'min_child_weight': 7, 'reg_alpha': 0.5, 'reg_lambda': 3.0,
            'gamma': 0.5, 'subsample': 0.8, 'colsample_bytree': 0.8,
        },
        'rf_params': dict(RF_CONSERVATIVE),
        'param_grid': [
            {'max_depth': 3, 'n_estimators': 200, 'learning_rate': 0.03,
             'min_child_weight': 7},
            {'max_depth': 2, 'n_estimators': 150, 'learning_rate': 0.03,
             'min_child_weight': 10},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES + LAG_FEATURES_FULL
            + ROLLING_FEATURES_SHORT + ENV_FEATURES_CORE
            + TEMPORAL_FEATURES_CORE
        ),
        'ensemble_weights': (0.10, 0.85, 0.05),
        'prediction_clip_q': 0.97,
        'prediction_clip_max': None,
    },

    # ==================================================================
    # STRUGGLE SITES
    # ==================================================================

    'Cannon Beach': {
        # Leak-free: N=61, XGB=-0.471, RF=-0.521, Naive=-10.663, Ens=-0.527.
        'xgb_params': {
            'max_depth': 2, 'n_estimators': 100, 'learning_rate': 0.03,
            'min_child_weight': 10, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
            'gamma': 1.0, 'subsample': 0.7, 'colsample_bytree': 0.7,
        },
        'rf_params': dict(RF_CONSERVATIVE),
        'param_grid': [
            {'max_depth': 2, 'n_estimators': 80, 'learning_rate': 0.02,
             'min_child_weight': 12},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES + LAG_FEATURES_SHORT
            + TEMPORAL_FEATURES_CORE + ['modis-sst', 'pdo']
        ),
        'ensemble_weights': (0.95, 0.03, 0.02),
        'prediction_clip_q': 0.95,
        'prediction_clip_max': 80.0,
    },

    'Gold Beach': {
        # Leak-free: N=144, XGB=-0.136, RF=-0.100, Naive=-1.656, Ens=-0.129.
        'xgb_params': {
            'max_depth': 2, 'n_estimators': 150, 'learning_rate': 0.03,
            'min_child_weight': 10, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
            'gamma': 1.0, 'subsample': 0.7, 'colsample_bytree': 0.7,
        },
        'rf_params': dict(RF_CONSERVATIVE),
        'param_grid': [
            {'max_depth': 2, 'n_estimators': 150, 'learning_rate': 0.03,
             'min_child_weight': 10},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES + LAG_FEATURES_SHORT
            + ROLLING_FEATURES_SHORT + ['modis-sst', 'pdo']
            + TEMPORAL_FEATURES_CORE
        ),
        'ensemble_weights': (0.40, 0.57, 0.03),
        'prediction_clip_q': 0.95,
        'prediction_clip_max': None,
    },

    'Newport': {
        # Leak-free: N=142, XGB=-0.051, RF=-0.011, Naive=-0.287, Ens=-0.038.
        'xgb_params': {
            'max_depth': 3, 'n_estimators': 250, 'learning_rate': 0.03,
            'min_child_weight': 7, 'reg_alpha': 0.5, 'reg_lambda': 3.0,
            'gamma': 0.5, 'subsample': 0.8, 'colsample_bytree': 0.8,
        },
        'rf_params': dict(RF_CONSERVATIVE),
        'param_grid': [
            {'max_depth': 3, 'n_estimators': 250, 'learning_rate': 0.03,
             'min_child_weight': 7},
            {'max_depth': 4, 'n_estimators': 200, 'learning_rate': 0.03,
             'min_child_weight': 5},
            {'max_depth': 2, 'n_estimators': 150, 'learning_rate': 0.03,
             'min_child_weight': 10},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES + LAG_FEATURES_FULL
            + ROLLING_FEATURES_SHORT + ENV_FEATURES_CORE
            + TEMPORAL_FEATURES_CORE
        ),
        'ensemble_weights': (0.25, 0.65, 0.10),
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

    Default: (0.30, 0.50, 0.20).  RF is globally the strongest model
    in the leak-free pipeline.
    """
    weights = get_site_config(site)['ensemble_weights']
    if weights is not None:
        return weights
    return (0.30, 0.50, 0.20)


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

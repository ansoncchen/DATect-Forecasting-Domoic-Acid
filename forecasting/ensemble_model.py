"""
Ensemble Model Utilities
========================

Provides ensemble regression models combining multiple tree-based learners.
"""

from sklearn.ensemble import VotingRegressor

import config

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def build_regression_ensemble():
    """
    Build a VotingRegressor with XGBoost, LightGBM, and CatBoost.
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")
    if not HAS_CATBOOST:
        raise ImportError("CatBoost not installed. Run: pip install catboost")

    xgb_params = getattr(config, "XGB_REGRESSION_PARAMS", {})
    lgbm_params = getattr(config, "LGBM_REGRESSION_PARAMS", {})
    cat_params = getattr(config, "CATBOOST_REGRESSION_PARAMS", {})

    models = [
        ("xgb", xgb.XGBRegressor(**xgb_params, n_jobs=-1, random_state=config.RANDOM_SEED)),
        ("lgbm", LGBMRegressor(**lgbm_params)),
        ("cat", CatBoostRegressor(**cat_params)),
    ]

    return VotingRegressor(models)

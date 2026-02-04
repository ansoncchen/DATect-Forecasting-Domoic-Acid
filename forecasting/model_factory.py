"""
Model Factory
=============

Creates and configures machine learning models for DA forecasting.
Supports both regression and classification tasks with multiple algorithms.

GPU Support:
- XGBoost uses tree_method='gpu_hist' when USE_GPU is True or auto-detected.
- Fallback to 'hist' (CPU) when GPU unavailable or USE_GPU=False.
"""

import os

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import config
from .logging_config import get_logger

logger = get_logger(__name__)


def _xgboost_tree_method():
    """
    Determine tree_method for XGBoost: gpu_hist when GPU available, else hist.
    Respects config.USE_GPU: True=force GPU, False=force CPU, None=auto-detect.
    """
    use_gpu = getattr(config, "USE_GPU", None)
    if use_gpu is False:
        return "hist"
    if use_gpu is True:
        return "gpu_hist"
    # Auto-detect via nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            return "gpu_hist"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "hist"


class ModelFactory:
    """
    Factory class for creating configured ML models.
    
    Supported Models:
    - XGBoost (regression & classification) - PRIMARY MODEL
    - Linear models (Linear/Logistic) - ALTERNATIVE MODEL
    - Linear Regression (regression)
    - Logistic Regression (classification)
    """
    
    def __init__(self):
        self.random_seed = config.RANDOM_SEED
        
    def get_model(self, task, model_type, params_override=None):
        if task == "regression":
            return self._get_regression_model(model_type, params_override=params_override)
        elif task == "classification":
            return self._get_classification_model(model_type, params_override=params_override)
        else:
            raise ValueError(f"Unknown task: {task}. Must be 'regression' or 'classification'")
            
    def _get_regression_model(self, model_type, params_override=None):
        if model_type == "xgboost" or model_type == "xgb":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
            default_reg_params = {
                'n_estimators': 400,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'colsample_bylevel': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'gamma': 0.1,
                'min_child_weight': 3,
                'tree_method': 'hist',
                'random_state': self.random_seed,
                'n_jobs': -1,
            }
            cfg_params = getattr(config, 'XGB_REGRESSION_PARAMS', None)
            params = {**default_reg_params, **(cfg_params or {})}
            if params_override:
                params.update(params_override)
            # GPU: use gpu_hist when available unless explicitly overridden
            if not (params_override and 'tree_method' in params_override):
                params['tree_method'] = _xgboost_tree_method()
            params['random_state'] = self.random_seed
            params['n_jobs'] = -1
            logger.debug(f"XGBoost regression tree_method={params['tree_method']}")
            return xgb.XGBRegressor(**params)
        elif model_type == "linear":
            return LinearRegression(
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown regression model: {model_type}. "
                           f"Supported: 'xgboost', 'linear')")
            
    def _get_classification_model(self, model_type, params_override=None):
        if model_type == "xgboost" or model_type == "xgb":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
            default_cls_params = {
                'n_estimators': 500,
                'max_depth': 7,
                'learning_rate': 0.03,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'colsample_bylevel': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 2.0,
                'gamma': 0.2,
                'min_child_weight': 5,
                'tree_method': 'hist',
                'random_state': self.random_seed,
                'n_jobs': -1,
                'eval_metric': 'logloss',
            }
            cfg_params = getattr(config, 'XGB_CLASSIFICATION_PARAMS', None)
            params = {**default_cls_params, **(cfg_params or {})}
            if params_override:
                params.update(params_override)
            # GPU: use gpu_hist when available unless explicitly overridden
            if not (params_override and 'tree_method' in params_override):
                params['tree_method'] = _xgboost_tree_method()
            params['random_state'] = self.random_seed
            params['n_jobs'] = -1
            logger.debug(f"XGBoost classification tree_method={params['tree_method']}")
            return xgb.XGBClassifier(**params)
        elif model_type == "logistic":
            return LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
                random_state=self.random_seed,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown classification model: {model_type}. "
                           f"Supported: 'xgboost', 'logistic')")
            
    def get_supported_models(self, task=None):
        models = {
            "regression": ["xgboost", "linear"],
            "classification": ["xgboost", "logistic"]
        }
        
        if task is None:
            return models
        elif task in models:
            return {task: models[task]}
        else:
            raise ValueError(f"Unknown task: {task}")
            
    def get_model_description(self, model_type):
        descriptions = {
            "xgboost": "XGBoost",
            "xgb": "XGBoost", 
            "linear": "Linear Regression",
            "logistic": "Logistic Regression"
        }
        
        return descriptions.get(model_type, f"Unknown model: {model_type}")
        
    def compute_sample_weights_for_classification(self, y_train):
        """
        Compute sample weights to handle class imbalance in classification.
        Returns weights that emphasize minority classes (especially extreme events).
        """
        import numpy as np
        
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
        
        # Create mapping of class to weight
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        # Apply weights to each sample
        sample_weights = np.array([class_weight_dict[y] for y in y_train])
        
        # Use balanced class weights only - no additional hardcoded modifiers
        # to ensure consistent and configurable weighting system
        return sample_weights
        
    def compute_spike_focused_weights(self, y_actual):
        """
        Compute sample weights specifically for spike detection timing.
        Heavily penalizes false negatives (missed spikes) and moderately penalizes false positives.
        """
        import numpy as np
        
        # Convert DA values to binary spike indicators
        actual_spikes = (y_actual > config.SPIKE_THRESHOLD).astype(int)
        
        # Initialize weights
        sample_weights = np.ones(len(actual_spikes))
        
        # Spike events get massive weight (focus on not missing these)
        spike_mask = actual_spikes == 1
        sample_weights[spike_mask] = config.SPIKE_FALSE_NEGATIVE_WEIGHT
        
        # Non-spike events get minimal weight (most of the year)
        non_spike_mask = actual_spikes == 0
        sample_weights[non_spike_mask] = config.SPIKE_TRUE_NEGATIVE_WEIGHT
        
        spike_count = spike_mask.sum()
        total_count = len(actual_spikes)
        
        logger.debug(f"Spike-focused weights: {spike_count}/{total_count} spikes with weight {config.SPIKE_FALSE_NEGATIVE_WEIGHT}")
        logger.debug(f"Non-spike samples: {total_count-spike_count} with weight {config.SPIKE_TRUE_NEGATIVE_WEIGHT}")
        
        return sample_weights

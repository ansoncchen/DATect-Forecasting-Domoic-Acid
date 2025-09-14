"""
Model Factory
=============

Creates and configures machine learning models for DA forecasting.
Supports both regression and classification tasks with multiple algorithms.
"""

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
            try:
                # Use XGBoost Random Forest instead of gradient boosting
                from xgboost import XGBRFRegressor
                default_reg_params = {
                    'n_estimators': 200,  # Number of trees in the forest
                    'max_depth': 8,
                    'subsample': 0.8,
                    'colsample_bynode': 0.8,
                    'reg_alpha': 0.01,
                    'reg_lambda': 0.1,
                    'min_child_weight': 1,
                    'random_state': self.random_seed,
                    'n_jobs': -1,
                }
                cfg_params = getattr(config, 'XGB_REGRESSION_PARAMS', None)
                params = {**default_reg_params, **(cfg_params or {})}
                if params_override:
                    params.update(params_override)
                params['random_state'] = self.random_seed
                params['n_jobs'] = -1
                return XGBRFRegressor(**params)
            except ImportError:
                # Fallback to manual Random Forest configuration for older XGBoost
                logger.warning("XGBRFRegressor not available. Using manual Random Forest configuration.")
                default_reg_params = {
                    'n_estimators': 1,  # Must be 1 for RF (use num_parallel_tree instead)
                    'num_parallel_tree': 200,  # Number of trees in the forest
                    'max_depth': 8,
                    'learning_rate': 1,  # Must be 1 for RF regression
                    'subsample': 0.8,
                    'colsample_bynode': 0.8,
                    'reg_alpha': 0.01,
                    'reg_lambda': 0.1,
                    'min_child_weight': 1,
                    'tree_method': 'hist',
                    'random_state': self.random_seed,
                    'n_jobs': -1,
                }
                cfg_params = getattr(config, 'XGB_REGRESSION_PARAMS', None)
                params = {**default_reg_params, **(cfg_params or {})}
                if params_override:
                    params.update(params_override)
                params['random_state'] = self.random_seed
                params['n_jobs'] = -1
                params['learning_rate'] = 1  # Enforce for RF
                params['n_estimators'] = 1  # Enforce for RF
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
            try:
                # Use XGBoost Random Forest instead of gradient boosting
                from xgboost import XGBRFClassifier
                default_cls_params = {
                    'n_estimators': 100,  # Number of trees in the forest
                    'max_depth': 7,
                    'subsample': 0.8,
                    'colsample_bynode': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 2.0,
                    'min_child_weight': 5,
                    'random_state': self.random_seed,
                    'n_jobs': -1,
                    'eval_metric': 'logloss',
                }
                cfg_params = getattr(config, 'XGB_CLASSIFICATION_PARAMS', None)
                params = {**default_cls_params, **(cfg_params or {})}
                if params_override:
                    params.update(params_override)
                params['random_state'] = self.random_seed
                params['n_jobs'] = -1
                return XGBRFClassifier(**params)
            except ImportError:
                # Fallback to manual Random Forest configuration for older XGBoost
                logger.warning("XGBRFClassifier not available. Using manual Random Forest configuration.")
                default_cls_params = {
                    'n_estimators': 1,  # Must be 1 for RF (use num_parallel_tree instead)
                    'num_parallel_tree': 100,  # Number of trees in the forest
                    'max_depth': 7,
                    'learning_rate': 1,  # Must be 1 for RF
                    'subsample': 0.8,
                    'colsample_bynode': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 2.0,
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
                params['random_state'] = self.random_seed
                params['n_jobs'] = -1
                params['learning_rate'] = 1  # Enforce for RF
                params['n_estimators'] = 1  # Enforce for RF
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
            "xgboost": "XGBoost Random Forest",
            "xgb": "XGBoost Random Forest",
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

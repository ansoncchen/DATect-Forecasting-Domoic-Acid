"""
Leak-Free Forecasting Engine
Core forecasting with complete temporal integrity protection
"""

import os
import pandas as pd
import numpy as np
import warnings
import random
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score

import config
from .data_processor import DataProcessor
from .model_factory import ModelFactory
from .validation import validate_system_startup, validate_runtime_parameters
from .logging_config import get_logger

warnings.filterwarnings('ignore')

logger = get_logger(__name__)


class ForecastEngine:
    """
    Leak-free domoic acid forecasting engine with temporal integrity.
    
    Features: Per-forecast DA categories, strict temporal ordering,
    temporal buffers for all features, no future data leakage.
    """
    
    def __init__(self, data_file=None, validate_on_init=True):
        logger.info("Initializing ForecastEngine")
        self.data_file = data_file or config.FINAL_OUTPUT_PATH
        self.data = None
        self.results_df = None
        self._data_cache = {}  # Cache for loaded data
        self._model_cache = {}  # Cache for trained models
        
        logger.info(f"Using data file: {self.data_file}")
        
        if validate_on_init:
            logger.info("Validating system startup configuration")
            validate_system_startup()
            logger.info("System startup validation completed successfully")
        
        logger.info("Initializing data processor and model factory")
        self.data_processor = DataProcessor()
        self.model_factory = ModelFactory()
        
        
        self.min_training_samples = max(1, int(getattr(config, 'MIN_TRAINING_SAMPLES', 5)))
        self.random_seed = config.RANDOM_SEED
        
        logger.info(f"Configuration: min_samples={self.min_training_samples}, seed={self.random_seed}")
        
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        logger.info("ForecastEngine initialization completed successfully")
        
    def run_retrospective_evaluation(self, task="regression", model_type="xgboost", 
                                   n_anchors=50, min_test_date="2008-01-01", 
                                   model_params_override=None):
        """
        Run leak-free retrospective evaluation matching original behavior.
        
        Args:
            task: "regression" or "classification"
            model_type: "xgboost", "linear", or "logistic" 
            n_anchors: Number of random anchor points per site
            min_test_date: Earliest date for test anchors
            
        Returns:
            DataFrame with evaluation results matching original format
        """
        validate_runtime_parameters(n_anchors, min_test_date)
        
        logger.info(f"Running LEAK-FREE {task} evaluation with {model_type}")
        
        # Use cached data if available
        cache_key = f"data_{self.data_file}"
        if cache_key not in self._data_cache:
            logger.info("Loading data into cache")
            self._data_cache[cache_key] = self.data_processor.load_and_prepare_base_data(self.data_file)
        else:
            logger.info("Using cached data")
        self.data = self._data_cache[cache_key]
        min_target_date = pd.Timestamp(min_test_date)
        
        self.last_diagnostics = {
            "task": task,
            "model_type": model_type,
            "min_test_date": str(min_test_date),
            "per_site": {}
        }

        anchor_infos = []
        for site in self.data["site"].unique():
            self.last_diagnostics["per_site"][site] = {
                "candidate_dates": 0,
                "valid_future": 0,
                "selected": 0,
                "earliest_selected_date": None
            }
            site_dates = self.data[self.data["site"] == site]["date"].sort_values().unique()
            # Need enough data span to support the forecast horizon
            if len(site_dates) > 1:
                date_span_days = (site_dates[-1] - site_dates[0]).days
                if date_span_days >= config.FORECAST_HORIZON_DAYS * 2:  # At least 2x horizon for meaningful evaluation
                    # Only use dates that have sufficient history and future data
                    valid_anchors = []
                    for i, date in enumerate(site_dates[:-1]):  # Exclude last date
                        self.last_diagnostics["per_site"][site]["candidate_dates"] += 1
                        if date >= min_target_date:
                            # Check if there's a future date at the required forecast horizon
                            future_dates = site_dates[i+1:]
                            valid_future = [d for d in future_dates if (d - date).days >= config.FORECAST_HORIZON_DAYS]
                            if valid_future:
                                self.last_diagnostics["per_site"][site]["valid_future"] += 1
                                valid_anchors.append(date)
                    
                    if valid_anchors:
                        n_sample = min(len(valid_anchors), n_anchors)
                        # Restore random sampling of anchors to avoid selection bias
                        selected_anchors = random.sample(list(valid_anchors), n_sample)
                        anchor_infos.extend([(site, pd.Timestamp(d)) for d in selected_anchors])
                        sel_sorted = sorted(selected_anchors)
                        self.last_diagnostics["per_site"][site]["selected"] = len(selected_anchors)
                        self.last_diagnostics["per_site"][site]["earliest_selected_date"] = str(sel_sorted[0].date()) if sel_sorted else None
        
        if not anchor_infos:
            logger.warning("No valid anchor points generated")
            return None
        
        logger.info(f"Generated {len(anchor_infos)} leak-free anchor points")
        
        n_jobs = int(os.getenv("DATECT_RETRO_N_JOBS", getattr(config, "RETROSPECTIVE_N_JOBS", -1)))
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(self._forecast_single_anchor_leak_free)(ai, self.data, min_target_date, task, model_type, model_params_override) 
            for ai in tqdm(anchor_infos, desc="Processing Leak-Free Anchors")
        )
        
        forecast_dfs = [df for df in results if df is not None]
        if not forecast_dfs:
            logger.warning("No successful forecasts")
            return None
            
        final_df = pd.concat(forecast_dfs, ignore_index=True)
        final_df = final_df.sort_values(["date", "site"]).drop_duplicates(["date", "site"])
        logger.info(f"Successfully processed {len(forecast_dfs)} leak-free forecasts")
        
        self.results_df = final_df
        
        self._display_evaluation_metrics(task)
        
        return final_df
        
    def _forecast_single_anchor_leak_free(self, anchor_info, full_data, min_target_date, task, model_type, model_params_override=None):
        """Process single anchor forecast with ZERO data leakage - original algorithm."""
        site, anchor_date = anchor_info
        
        site_data = full_data[full_data["site"] == site].copy()
        site_data.sort_values("date", inplace=True)
        
        train_mask = site_data["date"] <= anchor_date
        
        # Calculate target forecast date based on configured horizon
        target_forecast_date = anchor_date + pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
        
        # Find test samples within reasonable range of target forecast date
        test_mask = (site_data["date"] > anchor_date) & (site_data["date"] >= min_target_date)
        
        train_df = site_data[train_mask].copy()
        test_candidates = site_data[test_mask]
        
        if train_df.empty or test_candidates.empty:
            return None
        
        # Find the test sample closest to the target forecast date
        test_candidates = test_candidates.copy()
        test_candidates['date_diff'] = abs((test_candidates['date'] - target_forecast_date).dt.days)
        closest_idx = test_candidates['date_diff'].idxmin()
        test_df = test_candidates.loc[[closest_idx]].copy()
        test_date = test_df["date"].iloc[0]
        
        site_data_with_lags = self.data_processor.create_lag_features_safe(
            site_data, "site", "da", config.LAG_FEATURES, anchor_date
        )

        if config.PN_LAG_FEATURES:
            if "pn" in site_data_with_lags.columns:
                site_data_with_lags = self.data_processor.create_lag_features_safe(
                    site_data_with_lags, "site", "pn", config.PN_LAG_FEATURES, anchor_date
                )

        if config.ENVIRONMENTAL_LAG_FEATURES:
            for feature_name, lags in config.ENVIRONMENTAL_LAG_FEATURES.items():
                if feature_name in site_data_with_lags.columns and lags:
                    site_data_with_lags = self.data_processor.create_lag_features_safe(
                        site_data_with_lags, "site", feature_name, lags, anchor_date
                    )

        if "pn" in site_data_with_lags.columns and "da_lag_1" in site_data_with_lags.columns:
            denom = site_data_with_lags["da_lag_1"].replace(0, np.nan)
            site_data_with_lags["pn_da_lag_ratio"] = site_data_with_lags["pn"] / denom
        
        train_df = site_data_with_lags[site_data_with_lags["date"] <= anchor_date].copy()
        test_df = site_data_with_lags[site_data_with_lags["date"] == test_date].copy()
        
        if train_df.empty or test_df.empty:
            return None
        
        train_df_clean = train_df.dropna(subset=["da"]).copy()
        if train_df_clean.empty or len(train_df_clean) < self.min_training_samples:
            return None
        
        train_df_clean["da-category"] = self.data_processor.create_da_categories_safe(train_df_clean["da"])
        train_df = train_df_clean
        
        base_drop_cols = ["date", "site", "da"]
        train_drop_cols = base_drop_cols + ["da-category"]
        test_drop_cols = base_drop_cols
        
        transformer, X_train = self.data_processor.create_numeric_transformer(train_df, train_drop_cols)
        X_test = test_df.drop(columns=[col for col in test_drop_cols if col in test_df.columns])
        
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
        self.data_processor.validate_transformer_temporal_safety(
            transformer, train_df, test_df, anchor_date
        )
        
        X_train_processed = transformer.fit_transform(X_train)
        X_test_processed = transformer.transform(X_test)
        
        if pd.isna(train_df["da"]).any():
            return None
        
        actual_da = test_df["da"].iloc[0] if not test_df["da"].isna().iloc[0] else None
        actual_category = self.data_processor.create_da_categories_safe(pd.Series([actual_da]))[0] if actual_da is not None else None
        
        result = {
            'date': test_date,
            'site': site,
            'anchor_date': anchor_date,
            'actual_da': actual_da,
            'actual_category': actual_category
        }
        
        if task == "regression" or task == "both":
            reg_model = self.model_factory.get_model("regression", model_type, params_override=model_params_override)
            
            y_train_raw = train_df["da"]
            y_train = np.log1p(y_train_raw) if config.USE_LOG_TARGET_TRANSFORM else y_train_raw
            
            if config.USE_REGRESSION_SAMPLE_WEIGHTS:
                sample_weights = self.model_factory.compute_spike_focused_weights(y_train_raw)
                try:
                    reg_model.fit(X_train_processed, y_train, sample_weight=sample_weights)
                except TypeError:
                    reg_model.fit(X_train_processed, y_train)
            else:
                reg_model.fit(X_train_processed, y_train)
            
            pred_da = reg_model.predict(X_test_processed)[0]
            if config.USE_LOG_TARGET_TRANSFORM:
                pred_da = np.expm1(pred_da)
            pred_da = max(0.0, float(pred_da))
            result['predicted_da'] = pred_da
        
        if task == "classification" or task == "both":
            unique_classes = train_df["da-category"].nunique()
            if unique_classes > 1:
                unique_cats = sorted(train_df["da-category"].unique())
                cat_mapping = {cat: i for i, cat in enumerate(unique_cats)}
                reverse_mapping = {i: cat for cat, i in cat_mapping.items()}
                
                y_train_encoded = train_df["da-category"].map(cat_mapping)
                
                cls_model = self.model_factory.get_model("classification", model_type, params_override=model_params_override)
                
                # Apply consistent class balancing for fair baseline comparison
                sample_weights_cls = self.model_factory.compute_sample_weights_for_classification(y_train_encoded)
                
                if model_type in ["xgboost", "xgb"]:
                    cls_model.fit(X_train_processed, y_train_encoded, sample_weight=sample_weights_cls)
                else:
                    try:
                        cls_model.fit(X_train_processed, y_train_encoded, sample_weight=sample_weights_cls)
                    except TypeError:
                        cls_model.fit(X_train_processed, y_train_encoded)
                pred_encoded = cls_model.predict(X_test_processed)[0]
                
                pred_category = reverse_mapping[pred_encoded]
                result['predicted_category'] = int(pred_category)
            else:
                dominant_class = train_df["da-category"].mode()[0]
                result['predicted_category'] = int(dominant_class)
                result['single_class_prediction'] = True
                
        
        return pd.DataFrame([result])
    
    def generate_bootstrap_confidence_intervals(self, X_train_processed, y_train, X_forecast, model_type, n_bootstrap=None):
        """
        Generate bootstrap confidence intervals using resampling.
        
        Args:
            X_train_processed: Processed training features
            y_train: Training targets
            X_forecast: Processed forecast features
            model_type: Type of model to use
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with quantile predictions
        """
        # Use config value if not specified
        if n_bootstrap is None:
            n_bootstrap = config.N_BOOTSTRAP_ITERATIONS
            
        predictions = []
        y_train_raw = y_train
        y_train_model = np.log1p(y_train_raw) if config.USE_LOG_TARGET_TRANSFORM else y_train_raw
        
        # Generate bootstrap predictions with subsampling for efficiency
        for _ in range(n_bootstrap):
            # Use proper bootstrap sampling with replacement for valid confidence intervals
            n_samples = len(X_train_processed)
            subsample_size = int(config.BOOTSTRAP_SUBSAMPLE_FRACTION * n_samples)
            bootstrap_indices = np.random.choice(n_samples, subsample_size, replace=True)
            
            # Handle both DataFrame and numpy array cases
            if hasattr(X_train_processed, 'iloc'):
                X_bootstrap = X_train_processed.iloc[bootstrap_indices]
                y_bootstrap = y_train_model.iloc[bootstrap_indices]
            else:
                X_bootstrap = X_train_processed[bootstrap_indices]
                if hasattr(y_train_model, '__getitem__'):
                    y_bootstrap = y_train_model[bootstrap_indices]
                else:
                    y_bootstrap = y_train_model.iloc[bootstrap_indices]
            
            # Train model on bootstrap sample
            bootstrap_model = self.model_factory.get_model("regression", model_type)
            
            # Apply consistent sample weighting strategy to bootstrap
            if config.USE_REGRESSION_SAMPLE_WEIGHTS:
                # Use sample weights in bootstrap for consistency with main model
                if hasattr(y_train_raw, 'iloc'):
                    raw_bootstrap = y_train_raw.iloc[bootstrap_indices]
                elif hasattr(y_train_raw, '__getitem__'):
                    raw_bootstrap = y_train_raw[bootstrap_indices]
                else:
                    raw_bootstrap = y_train_raw.iloc[bootstrap_indices]
                bootstrap_weights = self.model_factory.compute_spike_focused_weights(raw_bootstrap)
                try:
                    bootstrap_model.fit(X_bootstrap, y_bootstrap, sample_weight=bootstrap_weights)
                except TypeError:
                    bootstrap_model.fit(X_bootstrap, y_bootstrap)
            else:
                # No sample weights for fair baseline comparison (current default)
                bootstrap_model.fit(X_bootstrap, y_bootstrap)
            
            # Make prediction
            pred = bootstrap_model.predict(X_forecast)[0]
            if config.USE_LOG_TARGET_TRANSFORM:
                pred = np.expm1(pred)
            pred = max(0.0, float(pred))
            predictions.append(pred)
        
        # Calculate quantiles
        predictions = np.array(predictions)
        return {
            "q05": float(np.percentile(predictions, config.CONFIDENCE_PERCENTILES[0])),
            "q50": float(np.percentile(predictions, config.CONFIDENCE_PERCENTILES[1])),
            "q95": float(np.percentile(predictions, config.CONFIDENCE_PERCENTILES[2])),
            "bootstrap_predictions": predictions.tolist()
        }

    def generate_single_forecast(self, data_path, forecast_date, site, task, model_type):
        """
        Generate a single forecast for a specific date and site using original algorithm.
        
        Args:
            data_path: Path to data file
            forecast_date: Date to forecast for
            site: Site to forecast for
            task: "regression" or "classification"
            model_type: Model type to use
            
        Returns:
            Dictionary with forecast results or None if insufficient data
        """
        # Use cached data if available
        cache_key = f"data_{data_path}"
        if cache_key not in self._data_cache:
            self._data_cache[cache_key] = self.data_processor.load_and_prepare_base_data(data_path)
        data = self._data_cache[cache_key]
        forecast_date = pd.Timestamp(forecast_date)
        
        self.data_processor.validate_forecast_inputs(data, site, forecast_date)
        
        df_site = data[data['site'] == site].copy()
        df_site.sort_values('date', inplace=True)
        
        # Calculate target anchor date based on forecast horizon
        target_anchor_date = forecast_date - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
        
        available_before = df_site[df_site['date'] < forecast_date]
        if available_before.empty:
            return None
        
        # Find the available data point closest to our target anchor date    
        available_before = available_before.copy()
        available_before['anchor_diff'] = abs((available_before['date'] - target_anchor_date).dt.days)
        closest_idx = available_before['anchor_diff'].idxmin()
        anchor_date = available_before.loc[closest_idx, 'date']
        
        df_site_with_lags = self.data_processor.create_lag_features_safe(
            df_site, "site", "da", config.LAG_FEATURES, anchor_date
        )

        if config.PN_LAG_FEATURES:
            if "pn" in df_site_with_lags.columns:
                df_site_with_lags = self.data_processor.create_lag_features_safe(
                    df_site_with_lags, "site", "pn", config.PN_LAG_FEATURES, anchor_date
                )

        if config.ENVIRONMENTAL_LAG_FEATURES:
            for feature_name, lags in config.ENVIRONMENTAL_LAG_FEATURES.items():
                if feature_name in df_site_with_lags.columns and lags:
                    df_site_with_lags = self.data_processor.create_lag_features_safe(
                        df_site_with_lags, "site", feature_name, lags, anchor_date
                    )

        if "pn" in df_site_with_lags.columns and "da_lag_1" in df_site_with_lags.columns:
            denom = df_site_with_lags["da_lag_1"].replace(0, np.nan)
            df_site_with_lags["pn_da_lag_ratio"] = df_site_with_lags["pn"] / denom
        
        df_train = df_site_with_lags[df_site_with_lags['date'] <= anchor_date].copy()
        df_train_clean = df_train.dropna(subset=['da']).copy()
        
        if df_train_clean.empty or len(df_train_clean) < self.min_training_samples:
            return None
        
        df_train_clean["da-category"] = self.data_processor.create_da_categories_safe(df_train_clean["da"])
        
        drop_cols = ["date", "site", "da", "da-category"]
        transformer, X_train = self.data_processor.create_numeric_transformer(df_train_clean, drop_cols)
        
        # CRITICAL: Validate temporal safety (single forecast uses all training data)
        if 'date' in df_train_clean.columns:
            future_data = df_train_clean[df_train_clean['date'] > anchor_date]
            if not future_data.empty:
                logger.error(f"TEMPORAL LEAKAGE in single forecast: {len(future_data)} records after anchor")
                raise ValueError(f"Training data contains future data after {anchor_date}")
            logger.debug(f"Single forecast temporal safety: all training data ≤ {anchor_date}")
        
        X_train_processed = transformer.fit_transform(X_train)
        
        latest_data = df_train_clean.iloc[-1:].copy()
        X_forecast = transformer.transform(latest_data.drop(columns=drop_cols, errors='ignore'))
        
        result = {
            'forecast_date': forecast_date,
            'anchor_date': anchor_date,
            'site': site,
            'task': task,
            'model_type': model_type,
            'training_samples': len(df_train_clean)
        }
        
        if task == "regression":
            # Check model cache first
            model_cache_key = f"regression_{model_type}_{len(df_train_clean)}_{hash(str(df_train_clean['da'].values.tobytes()))}"
            if model_cache_key in self._model_cache:
                logger.debug("Using cached regression model")
                model = self._model_cache[model_cache_key]
            else:
                logger.debug("Training new regression model")
                model = self.model_factory.get_model("regression", model_type)
            
            y_train_raw = df_train_clean["da"]
            y_train = np.log1p(y_train_raw) if config.USE_LOG_TARGET_TRANSFORM else y_train_raw
            
            if model_cache_key not in self._model_cache:
                if config.USE_REGRESSION_SAMPLE_WEIGHTS:
                    sample_weights = self.model_factory.compute_spike_focused_weights(y_train_raw)
                    try:
                        model.fit(X_train_processed, y_train, sample_weight=sample_weights)
                    except TypeError:
                        model.fit(X_train_processed, y_train)
                else:
                    model.fit(X_train_processed, y_train)
                self._model_cache[model_cache_key] = model
            
            prediction = model.predict(X_forecast)[0]
            if config.USE_LOG_TARGET_TRANSFORM:
                prediction = np.expm1(prediction)
            prediction = max(0.0, float(prediction))
            result['predicted_da'] = prediction
            result['feature_importance'] = self.data_processor.get_feature_importance(model, X_train_processed.columns)
            
            # Generate bootstrap confidence intervals for regression tasks
            if len(df_train_clean) >= 5:
                bootstrap_quantiles = self.generate_bootstrap_confidence_intervals(
                    X_train_processed, y_train_raw, X_forecast, model_type
                )
                result['bootstrap_quantiles'] = bootstrap_quantiles
                logger.debug(f"Bootstrap confidence intervals: q05={bootstrap_quantiles['q05']:.3f}, q50={bootstrap_quantiles['q50']:.3f}, q95={bootstrap_quantiles['q95']:.3f}")
            
            logger.debug(f"Regression prediction completed for {site}: {prediction:.4f}")
            
        elif task == "classification":
            unique_classes = df_train_clean["da-category"].nunique()
            if unique_classes > 1:
                unique_cats = sorted(df_train_clean["da-category"].unique())
                cat_mapping = {cat: i for i, cat in enumerate(unique_cats)}
                reverse_mapping = {i: cat for cat, i in cat_mapping.items()}
                
                y_train_encoded = df_train_clean["da-category"].map(cat_mapping)
                
                # Check model cache for classification
                model_cache_key = f"classification_{model_type}_{len(df_train_clean)}_{hash(str(df_train_clean['da-category'].values))}"
                if model_cache_key in self._model_cache:
                    logger.debug("Using cached classification model")
                    model = self._model_cache[model_cache_key]
                else:
                    logger.debug("Training new classification model")
                    model = self.model_factory.get_model("classification", model_type)
                if model_cache_key not in self._model_cache:
                    # Apply consistent class balancing for fair baseline comparison
                    sample_weights_cls = self.model_factory.compute_sample_weights_for_classification(y_train_encoded)
                    
                    if model_type in ["xgboost", "xgb"]:
                        model.fit(X_train_processed, y_train_encoded, sample_weight=sample_weights_cls)
                    else:
                        # Linear models should also use class balancing for fair comparison
                        try:
                            model.fit(X_train_processed, y_train_encoded, sample_weight=sample_weights_cls)
                        except TypeError:
                            # Fallback if model doesn't support sample_weight
                            model.fit(X_train_processed, y_train_encoded)
                    self._model_cache[model_cache_key] = model
                pred_encoded = model.predict(X_forecast)[0]
                
                prediction = reverse_mapping[pred_encoded]
                result['predicted_category'] = int(prediction)
                result['feature_importance'] = self.data_processor.get_feature_importance(model, X_train_processed.columns)
                logger.debug(f"Classification prediction completed for {site}: {prediction}")
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_forecast)[0]
                    prob_array = [0.0, 0.0, 0.0, 0.0]
                    for i, prob in enumerate(probabilities):
                        original_cat = reverse_mapping[i]
                        prob_array[original_cat] = float(prob)
                    result['class_probabilities'] = prob_array
                        
            else:
                dominant_class = df_train_clean["da-category"].mode()[0]
                result['predicted_category'] = int(dominant_class)
                result['single_class_prediction'] = True
                logger.debug(f"Single-class prediction for {site}: {dominant_class} (only class in training data)")
                
        return result
            
    def _display_evaluation_metrics(self, task):
        """Display evaluation metrics using original format."""
        if self.results_df is None or self.results_df.empty:
            logger.warning("No results for evaluation")
            return
            
        logger.info(f"Successfully processed {len(self.results_df)} forecasts")
        
        if task == "regression" or task == "both":
            valid_results = self.results_df.dropna(subset=['actual_da', 'predicted_da'])
            if not valid_results.empty:
                r2 = r2_score(valid_results['actual_da'], valid_results['predicted_da'])
                mae = mean_absolute_error(valid_results['actual_da'], valid_results['predicted_da'])
                
                spike_threshold = config.SPIKE_THRESHOLD
                y_true_binary = (valid_results['actual_da'] > spike_threshold).astype(int)
                y_pred_binary = (valid_results['predicted_da'] > spike_threshold).astype(int)
                
                precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                
                logger.info(f"Regression Metrics:")
                logger.info(f"  R2: {r2:.4f}, MAE: {mae:.4f}")
                logger.info(f"  Spike Detection (>{spike_threshold}): Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            else:
                logger.warning("No valid regression results for evaluation")
                
        if task == "classification" or task == "both":
            valid_results = self.results_df.dropna(subset=['actual_category', 'predicted_category'])
            if not valid_results.empty:
                y_true = valid_results['actual_category']
                y_pred = valid_results['predicted_category']
                
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                logger.info(f"Classification Metrics:")
                logger.info(f"  Accuracy: {accuracy:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1-Score: {f1:.4f}")
            else:
                logger.warning("No valid classification results for evaluation")
                
    

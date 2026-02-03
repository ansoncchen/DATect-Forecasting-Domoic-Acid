"""
Data Processing Module
=====================

Handles all data loading, cleaning, and feature engineering with temporal safeguards.
All operations maintain strict temporal integrity to prevent data leakage.
"""

import pandas as pd
import numpy as np
try:
    import duckdb
except ImportError:  # Optional dependency for faster parquet reads
    duckdb = None
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

import config
from .logging_config import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """
    Handles data processing with temporal safeguards.
    
    Key Features:
    - Forward-only interpolation
    - Temporal lag feature creation
    - Per-forecast DA category creation
    - Leak-free preprocessing pipelines
    """
    
    def __init__(self):
        logger.info("Initializing DataProcessor")
        self.da_category_bins = config.DA_CATEGORY_BINS
        self.da_category_labels = config.DA_CATEGORY_LABELS
        self._duckdb_conn = None
        self._duckdb_enabled = duckdb is not None
        logger.info(f"DA category configuration loaded: {len(self.da_category_bins)-1} categories")
        
    def validate_data_integrity(self, df, required_columns=None):
        """
        Validate data integrity for scientific forecasting.
        Keep only essential validation for temporal integrity.
        """
        logger.info("Starting data integrity validation")
        
        if required_columns is None:
            required_columns = ['date', 'site', 'da']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Critical columns missing: {missing_cols}")
            
        if 'da' in df.columns:
            da_values = df['da'].dropna()
            if not da_values.empty and (da_values < 0).any():
                raise ValueError("Negative DA values detected - invalid biological data")
                
        logger.info(f"Data integrity validation passed: {len(df)} records, {len(df.columns)} features")
        return True
        
    def validate_forecast_inputs(self, data, site, forecast_date):
        """
        Validate inputs for a specific forecast - CRITICAL for temporal integrity.
        """
        logger.info(f"Validating forecast inputs for {site} on {forecast_date}")
        
        # Validate site
        if site not in config.SITES:
            raise ValueError(f"Unknown site: '{site}'. Valid sites: {list(config.SITES.keys())}")
            
        forecast_date = pd.Timestamp(forecast_date)
        if forecast_date < pd.Timestamp('2000-01-01'):
            raise ValueError("Forecast date too early - satellite data not available before 2000")
        if forecast_date > pd.Timestamp.now() + pd.Timedelta(days=365):
            raise ValueError("Forecast date too far in future (>1 year)")
            
        site_data = data[data['site'] == site] if 'site' in data.columns else data
        if site_data.empty:
            raise ValueError(f"No historical data available for site: {site}")
            
        available_dates = site_data['date'].dropna()
        if available_dates.empty:
            raise ValueError(f"No valid dates in historical data for site: {site}")
            
        latest_data = available_dates.max()
        gap_days = (forecast_date - latest_data).days
        if gap_days > 365:
            logger.warning(f"Large gap between latest data ({latest_data.date()}) and forecast date ({forecast_date.date()}): {gap_days} days")
            
        logger.info(f"Forecast input validation passed for {site} on {forecast_date.date()}")
        return True
        
    def load_and_prepare_base_data(self, file_path):
        """
        Load base data WITHOUT any target-based preprocessing.
        """
        logger.info(f"Loading base data from {file_path}")

        if self._duckdb_enabled:
            if self._duckdb_conn is None:
                self._duckdb_conn = duckdb.connect(database=":memory:")
            data = self._duckdb_conn.execute(
                "SELECT * FROM read_parquet(?)",
                [file_path]
            ).fetchdf()
        else:
            data = pd.read_parquet(file_path, engine="pyarrow")
        logger.info(f"Raw data loaded: {len(data)} records, {len(data.columns)} columns")
        
        data["date"] = pd.to_datetime(data["date"])
        data.sort_values(["site", "date"], inplace=True)
        data.reset_index(drop=True, inplace=True)
        logger.debug("Data sorted and indexed by site and date")

        logger.info("Validating loaded data integrity")
        self.validate_data_integrity(data)

        logger.debug("Adding temporal features")
        
        if config.USE_ENHANCED_TEMPORAL_FEATURES:
            # Basic temporal encoding
            day_of_year = data["date"].dt.dayofyear
            data["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365)
            data["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365)
            
            # Enhanced temporal features for better accuracy
            data["month"] = data["date"].dt.month
            data["sin_month"] = np.sin(2 * np.pi * data["month"] / 12)
            data["cos_month"] = np.cos(2 * np.pi * data["month"] / 12)
            data["quarter"] = data["date"].dt.quarter
            data["days_since_start"] = (data["date"] - data["date"].min()).dt.days
            
            logger.debug("Enhanced temporal features added: sin/cos day_of_year, sin/cos month, quarter, days_since_start")
        else:
            # Minimal temporal features - just basic date components
            data["month"] = data["date"].dt.month
            data["quarter"] = data["date"].dt.quarter
            logger.debug("Basic temporal features added: month, quarter (enhanced features disabled)")

        # Add rolling statistics for key environmental features (temporal-safe)
        if config.USE_ROLLING_FEATURES:
            logger.debug("Adding rolling statistics features")
            data = self.add_rolling_statistics_safe(data)
        else:
            logger.debug("Rolling statistics features disabled by config")
        
        sites_count = data['site'].nunique()
        logger.info(f"Data preparation completed: {len(data)} records across {sites_count} sites")
        print(f"[INFO] Loaded {len(data)} records across {sites_count} sites")
        return data
        
    def create_lag_features_safe(self, df, group_col, value_col, lags, cutoff_date):
        """
        Create lag features with strict temporal cutoff to prevent leakage - CRITICAL for temporal integrity.
        """
        logger.info(f"Creating lag features for {value_col} with temporal cutoff at {cutoff_date}")
        logger.debug(f"Lag periods: {lags}")
        
        df = df.copy()
        df_sorted = df.sort_values([group_col, 'date'])
        
        created_features = []
        for lag in lags:
            feature_name = f"{value_col}_lag_{lag}"
            logger.debug(f"Creating lag feature: {feature_name}")
            
            df_sorted[feature_name] = df_sorted.groupby(group_col)[value_col].shift(lag)
            
            buffer_days = 1
            lag_cutoff_date = cutoff_date - pd.Timedelta(days=buffer_days)
            lag_cutoff_mask = df_sorted['date'] > lag_cutoff_date
            
            affected_rows = lag_cutoff_mask.sum()
            df_sorted.loc[lag_cutoff_mask, feature_name] = np.nan
            
            logger.debug(f"Lag feature {feature_name} created, {affected_rows} rows masked for temporal safety")
            created_features.append(feature_name)
            
        logger.info(f"Successfully created {len(created_features)} lag features with temporal safeguards")
        
        # Validate temporal integrity
        if df_sorted['date'].max() > cutoff_date:
            logger.debug(f"Data contains dates after cutoff ({cutoff_date}) - temporal leakage risk exists (expected for retrospective evaluation)")
            
        return df_sorted
        
    def add_rolling_statistics_safe(self, df):
        """
        Add rolling statistics features for better temporal pattern recognition.
        Focus on key oceanographic variables that influence DA blooms.
        """
        logger.info("Creating rolling statistics features")
        df = df.copy()
        df = df.sort_values(['site', 'date'])
        
        # Key environmental features to compute rolling stats for
        # Note: Each time step in final_output.parquet represents 1 week
        target_features = ['sst', 'chla', 'par', 'k490', 'fluorescence', 'pdo', 'oni', 'beuti', 'streamflow']
        windows = [2, 4, 8]  # 2-week, 4-week (monthly), 8-week (bi-monthly) patterns
        
        created_features = 0
        for feature in target_features:
            if feature in df.columns:
                for window in windows:
                    # Rolling mean
                    mean_col = f"{feature}_rolling_mean_{window}w"
                    df[mean_col] = df.groupby('site')[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    
                    # Rolling standard deviation (volatility)
                    std_col = f"{feature}_rolling_std_{window}w"
                    df[std_col] = df.groupby('site')[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                    
                    # Rolling trend (slope of linear regression)
                    trend_col = f"{feature}_rolling_trend_{window}w"
                    df[trend_col] = df.groupby('site')[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=2).apply(
                            lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= config.MIN_TREND_PERIODS else 0
                        )
                    )
                    
                    created_features += 3
                    
        logger.info(f"Created {created_features} rolling statistics features")
        return df
        
    def create_da_categories_safe(self, da_values):
        """
        Create DA categories from training data only.
        """
        logger.debug(f"Creating DA categories for {len(da_values)} values")
        
        if da_values.empty:
            logger.warning("Empty DA values provided for categorization")
            return pd.Series([], dtype=pd.Int64Dtype())
            
        invalid_mask = da_values < 0
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            logger.warning(f"Found {invalid_count} negative DA values - setting to NaN")
            da_values = da_values.copy()
            da_values[invalid_mask] = np.nan
            
        categories = pd.cut(
            da_values,
            bins=self.da_category_bins,
            labels=self.da_category_labels,
            right=True,
        ).astype(pd.Int64Dtype())
        
        value_counts = categories.value_counts().sort_index()
        logger.debug(f"DA category distribution: {dict(value_counts)}")
        
        return categories
        
    def create_numeric_transformer(self, df, drop_cols):
        """
        Create preprocessing transformer for numeric features.
        
        CRITICAL: This creates an UNFITTED transformer. The transformer must be
        fit ONLY on training data to prevent temporal leakage in retrospective tests.
        """
        logger.debug(f"Creating numeric transformer, dropping columns: {drop_cols}")
        
        X = df.drop(columns=drop_cols, errors="ignore")
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        logger.debug(f"Selected {len(numeric_cols)} numeric features: {list(numeric_cols)[:10]}...")
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for feature transformation")
            raise ValueError("No numeric features available for modeling")
        
        if 'date' in df.columns:
            date_range = df['date'].max() - df['date'].min()
            logger.debug(f"Transformer training data date range: {date_range.days} days "
                        f"({df['date'].min().date()} to {df['date'].max().date()})")
        
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),  # Fit only on training data
            ("scaler", MinMaxScaler()),                     # Fit only on training data
        ])
        
        transformer = ColumnTransformer(
            [("num", numeric_pipeline, numeric_cols)],
            remainder="drop",  # Drop non-numeric to avoid issues
            verbose_feature_names_out=False
        )
        transformer.set_output(transform="pandas")
                        
        logger.debug(f"Numeric transformer created (UNFITTED) with {len(numeric_cols)} features")
        logger.debug("CRITICAL: Transformer must be fit ONLY on temporal training data with anchor date validation!")
        
        return transformer, X
        
    def validate_transformer_temporal_safety(self, transformer, train_df, test_df, anchor_date):
        """
        Validate that transformer was fitted only on training data before anchor_date.
        
        This is CRITICAL for preventing temporal leakage in retrospective evaluation.
        """
        if train_df.empty or test_df.empty:
            return True
            
        # Check that training data is all before anchor date
        if 'date' in train_df.columns:
            future_training_data = train_df[train_df['date'] > anchor_date]
            if not future_training_data.empty:
                logger.error(f"TEMPORAL LEAKAGE: Training data contains {len(future_training_data)} "
                           f"records after anchor date {anchor_date}")
                raise ValueError(f"Training data contains future data after {anchor_date}")
        
        # Check that test data is after anchor date  
        if 'date' in test_df.columns:
            past_test_data = test_df[test_df['date'] <= anchor_date]
            if not past_test_data.empty:
                logger.error(f"TEMPORAL ERROR: Test data contains {len(past_test_data)} "
                           f"records before/on anchor date {anchor_date}")
                raise ValueError(f"Test data contains past data before/on {anchor_date}")
        
        logger.debug(f"Transformer temporal safety validated: training ≤ {anchor_date} < test")
        return True
        
    def validate_temporal_integrity(self, train_df, test_df):
        """
        Validate that temporal ordering is maintained - CRITICAL for preventing data leakage.
        """
        logger.debug("Validating temporal integrity between train and test sets")
        
        if train_df.empty or test_df.empty:
            logger.warning("Empty train or test dataframe - temporal validation failed")
            return False
            
        max_train_date = train_df['date'].max()
        min_test_date = test_df['date'].min()
        
        is_valid = max_train_date < min_test_date
        
        if is_valid:
            gap_days = (min_test_date - max_train_date).days
            logger.debug(f"Temporal integrity validated: {gap_days} day gap between train ({max_train_date.date()}) and test ({min_test_date.date()})")
        else:
            logger.error(f"TEMPORAL LEAKAGE DETECTED: train ends {max_train_date.date()}, test starts {min_test_date.date()}")
            raise ValueError(f"Training data ({max_train_date}) overlaps with test data ({min_test_date})")
            
        return is_valid
        
    def get_feature_importance(self, model, feature_names):
        """Extract basic feature importance from trained model."""
        if hasattr(model, 'feature_importances_'):
            return pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            return None
            
    def evaluate_spike_detection_performance(self, y_true, y_pred, y_pred_proba=None, threshold=None):
        """
        Evaluate spike detection performance with metrics focused on timing accuracy.
        """
        if threshold is None:
            threshold = config.SPIKE_THRESHOLD
            
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        import numpy as np
        
        # Convert to binary spike indicators if needed
        if not np.array_equal(np.unique(y_true), [0, 1]):
            y_true_binary = (y_true > threshold).astype(int)
        else:
            y_true_binary = y_true
            
        if not np.array_equal(np.unique(y_pred), [0, 1]):
            y_pred_binary = (y_pred > threshold).astype(int)
        else:
            y_pred_binary = y_pred
        
        # Core spike detection metrics
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # Confusion matrix for detailed analysis
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Spike-specific metrics
        spike_detection_rate = recall  # Same as recall but more intuitive name
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        missed_spike_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Total spikes in dataset
        total_spikes = np.sum(y_true_binary)
        total_samples = len(y_true_binary)
        spike_prevalence = total_spikes / total_samples if total_samples > 0 else 0
        
        metrics = {
            'spike_detection_rate': spike_detection_rate,  # % of actual spikes detected
            'false_alarm_rate': false_alarm_rate,          # % of non-spikes incorrectly flagged
            'missed_spike_rate': missed_spike_rate,        # % of actual spikes missed
            'precision': precision,                        # When we predict spike, how often correct?
            'recall': recall,                             # Of all actual spikes, how many detected?
            'f1_score': f1,                              # Harmonic mean of precision/recall
            'true_positives': int(tp),                    # Correctly detected spikes
            'false_positives': int(fp),                   # False alarms
            'false_negatives': int(fn),                   # Missed spikes
            'true_negatives': int(tn),                    # Correctly identified non-spikes
            'total_spikes': int(total_spikes),           # Actual spike events in dataset
            'total_samples': int(total_samples),          # Total predictions made
            'spike_prevalence': spike_prevalence          # % of samples that are spikes
        }
        
        return metrics
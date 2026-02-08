"""
Raw-Data Forecast Engine
========================

Drop-in replacement for the original ``ForecastEngine`` that uses the
raw-data ensemble pipeline (XGBoost + Random Forest + Naive baseline)
with per-site hyperparameter tuning, observation-order lag features,
leak-free anchor-date environmental features, and quantile prediction
intervals.

Provides the **exact same method signatures** as the old engine so the
FastAPI backend (``api.py``) works without interface changes.
"""

from __future__ import annotations

import os
import random
import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
    f1_score,
)
from tqdm import tqdm

import config
from .raw_data_forecaster import (
    load_raw_da_measurements,
    aggregate_raw_to_weekly,
    build_raw_feature_frame,
    get_last_known_raw_da,
    get_site_anchor_row,
    get_site_training_frame,
    recompute_test_row_persistence_features,
)
from .raw_model_factory import (
    build_xgb_regressor,
    build_rf_regressor,
    build_linear_regressor,
    build_logistic_classifier,
)
from .per_site_models import (
    apply_site_xgb_params,
    apply_site_rf_params,
    get_site_param_grid,
    get_site_ensemble_weights,
    get_site_clip_params,
    compute_site_drop_cols,
)
from .feature_utils import add_temporal_features, create_transformer
from .ensemble_model_factory import EnsembleModelFactory
from .classification_adapter import ClassificationAdapter
from .validation import validate_system_startup, validate_runtime_parameters
from .logging_config import get_logger

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: leakage verification (ported from source config.py)
# ---------------------------------------------------------------------------

def _verify_no_data_leakage(train_data, test_date, anchor_date):
    """Assert no training data leaks past the anchor date."""
    anchor = pd.Timestamp(anchor_date)
    if train_data["date"].max() > anchor:
        raise AssertionError(
            f"TEMPORAL LEAK: training data max date "
            f"{train_data['date'].max().date()} > anchor {anchor.date()}"
        )
    test = pd.Timestamp(test_date)
    if test <= anchor:
        raise AssertionError(
            f"TEMPORAL LEAK: test_date {test.date()} <= anchor {anchor.date()}"
        )


class RawForecastEngine:
    """
    Leak-free domoic acid forecasting engine using the raw-data ensemble
    pipeline.

    Replaces the original ``ForecastEngine`` while preserving the same
    public API (``generate_single_forecast``, ``run_retrospective_evaluation``,
    ``generate_bootstrap_confidence_intervals``).
    """

    def __init__(self, data_file=None, validate_on_init=True):
        logger.info("Initializing RawForecastEngine")
        self.data_file = data_file or config.FINAL_OUTPUT_PATH
        self.results_df = None
        self._data_cache: dict = {}
        self._feature_frame_cache: Optional[pd.DataFrame] = None
        self._raw_data_cache: Optional[pd.DataFrame] = None

        if validate_on_init:
            validate_system_startup()

        self.model_factory = EnsembleModelFactory()
        self.classification_adapter = ClassificationAdapter()

        self.min_training_samples = max(
            1, int(getattr(config, "MIN_TRAINING_SAMPLES", 10))
        )
        self.random_seed = config.RANDOM_SEED

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        logger.info("RawForecastEngine initialization completed")

    # ------------------------------------------------------------------
    # Data loading & caching
    # ------------------------------------------------------------------

    def _load_feature_frame(self) -> pd.DataFrame:
        """Build (or return cached) feature frame from raw + processed data."""
        if self._feature_frame_cache is not None:
            return self._feature_frame_cache

        logger.info("Building feature frame from raw + processed data")

        # Load processed environmental data
        processed = pd.read_parquet(self.data_file)
        processed["date"] = pd.to_datetime(processed["date"])
        processed = processed.sort_values(["site", "date"]).reset_index(drop=True)

        # Load raw DA measurements
        raw_df = load_raw_da_measurements()
        self._raw_data_cache = raw_df

        # Add temporal features to processed data before merging
        processed = add_temporal_features(processed)

        # Aggregate raw to weekly and build feature frame
        raw_weekly = aggregate_raw_to_weekly(raw_df)
        feature_frame = build_raw_feature_frame(processed, raw_weekly)

        self._feature_frame_cache = feature_frame
        logger.info(
            "Feature frame built: %d rows, %d columns",
            len(feature_frame),
            len(feature_frame.columns),
        )
        return feature_frame

    # ------------------------------------------------------------------
    # Single forecast (realtime)
    # ------------------------------------------------------------------

    def generate_single_forecast(
        self,
        data_path: str,
        forecast_date,
        site: str,
        task: str,
        model_type: str,
    ) -> Optional[dict]:
        """
        Generate a single forecast for a specific date and site.

        This is a faithful port of the source pipeline's
        ``run_single_raw_validation()`` logic, adapted for the API
        response contract.

        Parameters
        ----------
        data_path : str
            Path to processed parquet (used for cache key only).
        forecast_date : date-like
            Date to forecast for.
        site : str
            Monitoring site name.
        task : str
            ``"regression"`` or ``"classification"``.
        model_type : str
            ``"ensemble"``, ``"xgboost"``, ``"rf"``, ``"naive"``, etc.

        Returns
        -------
        dict or None
        """
        forecast_date = pd.Timestamp(forecast_date)

        # Load / cache feature frame
        feature_frame = self._load_feature_frame()

        # Anchor = forecast_date - 7 days
        anchor_date = forecast_date - pd.Timedelta(
            days=config.FORECAST_HORIZON_DAYS
        )

        # --- Training data ---
        train_data = get_site_training_frame(
            feature_frame, site, anchor_date, self.min_training_samples
        )
        if train_data is None:
            logger.warning(
                "Insufficient training data for %s at anchor %s",
                site,
                anchor_date.date(),
            )
            return None

        # --- Test row (leak-free: env features from anchor date) ---
        test_row = get_site_anchor_row(
            feature_frame, site, forecast_date, anchor_date,
            max_date_diff_days=28,
        )
        if test_row is None:
            logger.warning(
                "No anchor row available for %s at %s",
                site,
                anchor_date.date(),
            )
            return None

        # Recompute persistence features from training data only
        test_row = recompute_test_row_persistence_features(
            test_row, train_data, config.SPIKE_THRESHOLD
        )

        # Add temporal features
        train_data = add_temporal_features(train_data)
        test_row = add_temporal_features(test_row)

        # --- Feature preparation ---
        use_per_site = getattr(config, "USE_PER_SITE_MODELS", True)
        zero_imp = getattr(config, "ZERO_IMPORTANCE_FEATURES", [])

        drop_cols = ["date", "site", "da_raw", "da"] + list(zero_imp)

        if use_per_site:
            drop_cols = compute_site_drop_cols(
                drop_cols, train_data.columns.tolist(), site
            )

        try:
            transformer, X_train = create_transformer(train_data, drop_cols)
            y_train = train_data["da_raw"].astype(float).copy()

            X_train_processed = transformer.fit_transform(X_train)

            X_test = test_row.drop(columns=drop_cols, errors="ignore")
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
            X_test_processed = transformer.transform(X_test)

            # Leakage check
            _verify_no_data_leakage(train_data, forecast_date, anchor_date)

        except Exception as exc:
            logger.error("Feature preparation failed: %s", exc)
            return None

        # --- Prediction post-processing helper ---
        clip_q_global = getattr(config, "PREDICTION_CLIP_Q", 0.99)

        def _postprocess(value: float) -> float:
            value = max(0.0, value)
            if use_per_site:
                site_clip_q, site_clip_max = get_site_clip_params(site)
                cq = site_clip_q if site_clip_q is not None else clip_q_global
            else:
                cq = clip_q_global
                site_clip_max = None
            if cq is not None:
                clip_max = float(np.quantile(train_data["da_raw"], cq))
                value = min(value, clip_max)
            if site_clip_max is not None:
                value = min(value, site_clip_max)
            return float(value)

        # --- XGBoost prediction ---
        xgb_params = dict(config.XGB_REGRESSION_PARAMS)
        if use_per_site:
            xgb_params = apply_site_xgb_params(xgb_params, site)

        xgb_model = build_xgb_regressor(xgb_params)
        xgb_model.fit(X_train_processed, y_train)
        xgb_raw = float(xgb_model.predict(X_test_processed)[0])
        xgb_prediction = _postprocess(xgb_raw)

        # --- Random Forest prediction ---
        rf_params = dict(getattr(config, "RF_REGRESSION_PARAMS", {}))
        if use_per_site:
            rf_params = apply_site_rf_params(rf_params, site)
        rf_model = build_rf_regressor(rf_params)
        rf_model.fit(X_train_processed, y_train)
        rf_raw = float(rf_model.predict(X_test_processed)[0])
        rf_prediction = _postprocess(rf_raw)

        # --- Naive baseline ---
        naive_prediction = get_last_known_raw_da(
            train_data,
            anchor_date=anchor_date,
            max_age_days=getattr(config, "PERSISTENCE_MAX_DAYS", None),
        )
        if naive_prediction is None:
            naive_prediction = 0.0

        # --- Ridge (linear competitor) ---
        linear_prediction = None
        if model_type == "linear":
            try:
                linear_model = build_linear_regressor()
                linear_model.fit(X_train_processed, y_train)
                linear_raw = float(linear_model.predict(X_test_processed)[0])
                linear_prediction = _postprocess(linear_raw)
            except Exception as exc:
                logger.warning("Linear regression failed: %s", exc)
                linear_prediction = None

        # --- Ensemble blend ---
        w_xgb, w_rf, w_naive = get_site_ensemble_weights(site)
        ensemble_prediction = (
            w_xgb * xgb_prediction
            + w_rf * rf_prediction
            + w_naive * naive_prediction
        )

        # Choose which prediction to expose as primary
        if model_type in ("ensemble", "threshold"):
            primary_prediction = ensemble_prediction
        elif model_type in ("xgboost", "xgb"):
            primary_prediction = xgb_prediction
        elif model_type == "rf":
            primary_prediction = rf_prediction
        elif model_type == "naive":
            primary_prediction = naive_prediction
        elif model_type == "linear":
            primary_prediction = (
                linear_prediction
                if linear_prediction is not None
                else ensemble_prediction
            )
        else:
            primary_prediction = ensemble_prediction

        # --- Feature importance ---
        feature_importance = {}
        if hasattr(xgb_model, "feature_importances_"):
            try:
                feat_names = X_train.columns.tolist()
                feature_importance = dict(
                    zip(feat_names, xgb_model.feature_importances_)
                )
            except Exception:
                pass

        # Build result dict matching the API contract
        result = {
            "forecast_date": forecast_date,
            "anchor_date": anchor_date,
            "site": site,
            "task": task,
            "model_type": model_type,
            "training_samples": len(train_data),
            "predicted_da": float(primary_prediction),
            "feature_importance": feature_importance,
            # Ensemble breakdown (new fields)
            "naive_prediction": float(naive_prediction),
            "xgb_prediction": float(xgb_prediction),
            "rf_prediction": float(rf_prediction),
            "ensemble_prediction": float(ensemble_prediction),
            "ensemble_weights": [float(w_xgb), float(w_rf), float(w_naive)],
        }

        # --- Quantile / bootstrap confidence intervals ---
        if task == "regression":
            quantiles = self._compute_confidence_intervals(
                X_train_processed, y_train, X_test_processed, xgb_params,
                _postprocess, model_type, site, naive_prediction,
                rf_params, (w_xgb, w_rf, w_naive),
            )
            result["bootstrap_quantiles"] = quantiles

        # --- Classification ---
        if task == "classification" or task == "both":
            cls_result = self.classification_adapter.classify_prediction(
                primary_prediction
            )
            result["predicted_category"] = cls_result["predicted_category"]
            result["class_probabilities"] = cls_result["class_probabilities"]

            if model_type == "logistic":
                # Train LogisticRegression on category labels
                y_categories = self.classification_adapter.threshold_classify_series(
                    train_data["da_raw"]
                )
                unique_cats = sorted(y_categories.unique())
                if len(unique_cats) >= 2:
                    try:
                        logistic_model = build_logistic_classifier()
                        logistic_model.fit(X_train_processed, y_categories)
                        result["predicted_category"] = int(
                            logistic_model.predict(X_test_processed)[0]
                        )
                        if hasattr(logistic_model, "predict_proba"):
                            probs = logistic_model.predict_proba(X_test_processed)[0]
                            prob_array = [0.0, 0.0, 0.0, 0.0]
                            for i, cls in enumerate(logistic_model.classes_):
                                cls_id = int(cls)
                                if 0 <= cls_id <= 3:
                                    prob_array[cls_id] = float(probs[i])
                            result["class_probabilities"] = prob_array
                    except Exception as exc:
                        logger.debug("Logistic classifier failed: %s", exc)
                else:
                    result["predicted_category"] = (
                        self.classification_adapter.threshold_classify(primary_prediction)
                    )
            else:
                # Also try dedicated ML classifier for richer probabilities
                try:
                    classifier_result = (
                        self.classification_adapter.train_dedicated_classifier(
                            X_train_processed,
                            train_data["da_raw"],
                            site=site,
                        )
                    )
                    if classifier_result is not None:
                        full_cls = self.classification_adapter.classify_prediction(
                            primary_prediction, classifier_result, X_test_processed,
                        )
                        result["predicted_category_ml"] = full_cls["predicted_category_ml"]
                        result["class_probabilities"] = full_cls["class_probabilities"]
                except Exception as exc:
                    logger.debug("Dedicated classifier failed: %s", exc)
        elif task == "regression":
            # Even for regression, provide threshold classification for the UI
            result["predicted_category"] = (
                self.classification_adapter.threshold_classify(primary_prediction)
            )

        return result

    # ------------------------------------------------------------------
    # Confidence intervals
    # ------------------------------------------------------------------

    def _compute_confidence_intervals(
        self,
        X_train_processed,
        y_train,
        X_test_processed,
        xgb_params: dict,
        postprocess_fn,
        model_type: str,
        site: str,
        naive_prediction: float,
        rf_params: dict,
        ensemble_weights: tuple[float, float, float],
    ) -> dict:
        """
        Compute prediction intervals via XGBoost quantile objectives
        (preferred) with bootstrap fallback.
        """
        quantiles = {}
        enable_quantile = getattr(config, "ENABLE_QUANTILE_INTERVALS", True)
        enable_bootstrap = getattr(config, "ENABLE_BOOTSTRAP_INTERVALS", True)

        if enable_quantile and model_type in ("xgboost", "xgb", "ensemble", "threshold"):
            try:
                for q in (0.05, 0.50, 0.95):
                    # XGBoost quantile regression for the XGB component
                    q_params = {
                        **xgb_params,
                        "objective": "reg:quantile",
                        "quantile_alpha": q,
                    }
                    q_model = build_xgb_regressor(q_params)
                    q_model.fit(X_train_processed, y_train)
                    xgb_q_pred = postprocess_fn(float(q_model.predict(X_test_processed)[0]))

                    key = f"q{int(q * 100):02d}"
                    if model_type in ("ensemble", "threshold"):
                        # Blend XGB quantile with RF + naive for ensemble CI
                        rf_model = build_rf_regressor(rf_params)
                        rf_model.fit(X_train_processed, y_train)
                        rf_q_pred = postprocess_fn(float(rf_model.predict(X_test_processed)[0]))
                        w_xgb, w_rf, w_naive = ensemble_weights
                        quantiles[key] = (
                            w_xgb * xgb_q_pred
                            + w_rf * rf_q_pred
                            + w_naive * float(naive_prediction)
                        )
                    else:
                        quantiles[key] = xgb_q_pred
                return quantiles
            except Exception:
                logger.debug("Quantile objectives unavailable, falling back to bootstrap")

        if not enable_bootstrap:
            return {}

        # Bootstrap fallback
        return self.generate_bootstrap_confidence_intervals(
            X_train_processed, y_train, X_test_processed, model_type,
            postprocess_fn=postprocess_fn,
            naive_prediction=naive_prediction,
            rf_params=rf_params,
            ensemble_weights=ensemble_weights,
        )

    def generate_bootstrap_confidence_intervals(
        self,
        X_train_processed,
        y_train,
        X_forecast,
        model_type: str,
        n_bootstrap: Optional[int] = None,
        postprocess_fn=None,
        naive_prediction: Optional[float] = None,
        rf_params: Optional[dict] = None,
        ensemble_weights: Optional[tuple[float, float, float]] = None,
    ) -> dict:
        """
        Generate bootstrap confidence intervals using resampling.

        Preserved for backward compatibility with the old engine's interface.
        """
        if n_bootstrap is None:
            n_bootstrap = config.N_BOOTSTRAP_ITERATIONS

        predictions = []
        subsample_frac = getattr(config, "BOOTSTRAP_SUBSAMPLE_FRACTION", 0.75)
        if postprocess_fn is None:
            postprocess_fn = lambda val: max(0.0, float(val))

        rf_params = rf_params or dict(getattr(config, "RF_REGRESSION_PARAMS", {}))
        if ensemble_weights is None:
            ensemble_weights = (0.50, 0.15, 0.35)
        w_xgb, w_rf, w_naive = ensemble_weights
        if naive_prediction is None:
            naive_prediction = 0.0

        for _ in range(n_bootstrap):
            n_samples = len(X_train_processed)
            subsample_size = int(subsample_frac * n_samples)
            idx = np.random.choice(n_samples, subsample_size, replace=True)

            if hasattr(X_train_processed, "iloc"):
                X_bs = X_train_processed.iloc[idx]
                y_bs = y_train.iloc[idx]
            else:
                X_bs = X_train_processed[idx]
                y_bs = y_train[idx]

            if model_type in ("naive",):
                predictions.append(float(naive_prediction))
                continue

            if model_type in ("rf",):
                rf_model = build_rf_regressor(rf_params)
                rf_model.fit(X_bs, y_bs)
                rf_raw = float(rf_model.predict(X_forecast)[0])
                predictions.append(postprocess_fn(rf_raw))
                continue

            # Default to XGBoost-based bootstrap
            bs_model = self.model_factory.get_model(
                "regression", "xgboost", params_override=None,
            )
            if bs_model is None:
                bs_model = build_xgb_regressor()
            bs_model.fit(X_bs, y_bs)
            xgb_raw = float(bs_model.predict(X_forecast)[0])
            xgb_pred = postprocess_fn(xgb_raw)

            if model_type in ("ensemble", "threshold"):
                rf_model = build_rf_regressor(rf_params)
                rf_model.fit(X_bs, y_bs)
                rf_raw = float(rf_model.predict(X_forecast)[0])
                rf_pred = postprocess_fn(rf_raw)
                pred = (
                    w_xgb * xgb_pred
                    + w_rf * rf_pred
                    + w_naive * float(naive_prediction)
                )
                predictions.append(float(pred))
            elif model_type in ("linear",):
                linear_model = build_linear_regressor()
                linear_model.fit(X_bs, y_bs)
                linear_raw = float(linear_model.predict(X_forecast)[0])
                predictions.append(postprocess_fn(linear_raw))
            else:
                predictions.append(float(xgb_pred))

        predictions = np.array(predictions)
        percentiles = getattr(config, "CONFIDENCE_PERCENTILES", [5, 50, 95])
        return {
            "q05": float(np.percentile(predictions, percentiles[0])),
            "q50": float(np.percentile(predictions, percentiles[1])),
            "q95": float(np.percentile(predictions, percentiles[2])),
            "bootstrap_predictions": predictions.tolist(),
        }

    # ------------------------------------------------------------------
    # Retrospective evaluation
    # ------------------------------------------------------------------

    def run_retrospective_evaluation(
        self,
        task: str = "regression",
        model_type: str = "ensemble",
        n_anchors: int = 50,
        min_test_date: str = "2008-01-01",
        model_params_override: Optional[dict] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Run leak-free retrospective evaluation on raw DA measurements.

        Returns DataFrame with columns matching the API's expected
        canonical keys: ``date``, ``site``, ``anchor_date``, ``actual_da``,
        ``predicted_da``, ``actual_category``, ``predicted_category``.
        """
        validate_runtime_parameters(n_anchors, min_test_date)
        logger.info("Running LEAK-FREE %s evaluation with %s", task, model_type)
        self._retro_model_type = model_type

        feature_frame = self._load_feature_frame()
        raw_data = self._raw_data_cache

        min_test_ts = pd.Timestamp(
            getattr(config, "MIN_TEST_DATE", min_test_date)
        )
        min_training = self.min_training_samples
        forecast_horizon = config.FORECAST_HORIZON_DAYS
        history_frac = getattr(config, "HISTORY_REQUIREMENT_FRACTION", 0.33)

        # Filter to valid test dates
        candidate_raw = raw_data[raw_data["date"] >= min_test_ts].copy()
        site_total_counts = raw_data.groupby("site")["date"].size().to_dict()

        # Apply history requirement filter
        valid_rows = []
        for _, row in candidate_raw.iterrows():
            anchor_dt = row["date"] - pd.Timedelta(days=forecast_horizon)
            site = row["site"]
            total_site = site_total_counts.get(site, 0)
            if total_site == 0:
                continue
            min_required = max(
                int(np.ceil(history_frac * total_site)),
                min_training,
            )
            n_history = len(
                raw_data[
                    (raw_data["site"] == site) & (raw_data["date"] <= anchor_dt)
                ]
            )
            if n_history < min_required:
                continue
            site_history = feature_frame[
                (feature_frame["site"] == site)
                & (feature_frame["date"] <= anchor_dt)
                & (feature_frame["da_raw"].notna())
            ]
            if len(site_history) >= min_training:
                valid_rows.append(row)

        if not valid_rows:
            logger.warning("No valid test samples found")
            return None

        valid_df = pd.DataFrame(valid_rows)

        # Per-site sampling (~20% of total raw measurements)
        rng = np.random.RandomState(self.random_seed)
        sampled_rows = []
        for site, site_df in valid_df.groupby("site"):
            site_df = site_df.sort_values("date")
            n_candidates = len(site_df)
            total_site = site_total_counts.get(site, n_candidates)
            target = min(int(np.ceil(0.2 * total_site)), n_candidates)
            if target <= 0:
                continue
            idx = rng.choice(n_candidates, size=target, replace=False)
            sampled_rows.append(site_df.iloc[idx])

        if not sampled_rows:
            logger.warning("Per-site sampling produced no samples")
            return None

        test_samples = pd.concat(sampled_rows, ignore_index=True)
        logger.info("Selected %d test measurements for retrospective", len(test_samples))

        # XGBoost base parameters
        base_params = dict(config.XGB_REGRESSION_PARAMS)
        base_params["n_jobs"] = 1  # Avoid nested parallelism

        # Build measurement dicts
        sample_rows = [
            {"date": row["date"], "site": row["site"], "da_raw": row["da_raw"]}
            for _, row in test_samples.iterrows()
        ]

        # Run validation
        enable_parallel = getattr(config, "ENABLE_PARALLEL", True)
        n_jobs = getattr(config, "N_JOBS", -1)

        if enable_parallel:
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._run_single_retrospective)(
                    row, feature_frame, base_params
                )
                for row in tqdm(sample_rows, desc="Retrospective", unit="sample")
            )
        else:
            results = [
                self._run_single_retrospective(row, feature_frame, base_params)
                for row in tqdm(sample_rows, desc="Retrospective", unit="sample")
            ]

        n_processed = len(results)
        results = [r for r in results if r is not None]
        n_success = len(results)
        n_failed = n_processed - n_success

        if not results:
            logger.warning("No successful retrospective predictions")
            return None

        if n_failed > 0:
            logger.info(
                "Retrospective: %d/%d samples produced a prediction (%d failed: no train data, no test row, naive=None, or exception)",
                n_success, n_processed, n_failed,
            )

        results_df = pd.DataFrame(results)

        # --- Ensemble prediction ---
        use_per_site = getattr(config, "USE_PER_SITE_MODELS", True)
        if (
            not results_df.empty
            and "predicted_da" in results_df.columns
            and "naive_prediction" in results_df.columns
        ):
            if use_per_site:
                ens = []
                for _, row in results_df.iterrows():
                    w_xgb, w_rf, w_naive = get_site_ensemble_weights(row["site"])
                    rf_pred = row.get("predicted_da_rf", row["predicted_da"])
                    ens.append(
                        w_xgb * row["predicted_da"]
                        + w_rf * rf_pred
                        + w_naive * row["naive_prediction"]
                    )
                results_df["ensemble_prediction"] = ens
            else:
                w_xgb, w_rf, w_naive = 0.50, 0.15, 0.35
                rf_preds = results_df.get(
                    "predicted_da_rf", results_df["predicted_da"]
                )
                results_df["ensemble_prediction"] = (
                    w_xgb * results_df["predicted_da"]
                    + w_rf * rf_preds
                    + w_naive * results_df["naive_prediction"]
                )

        # --- Canonical keys for API ---
        # Map raw column names to what the API expects
        rename_map = {
            "actual_da_raw": "actual_da",
            "test_date": "date",
        }
        results_df = results_df.rename(columns=rename_map)

        # Derive categories
        if "actual_da" in results_df.columns:
            results_df["actual_category"] = (
                self.classification_adapter.threshold_classify_series(
                    results_df["actual_da"]
                )
            )
        if model_type == "naive":
            results_df["predicted_da"] = results_df["naive_prediction"]
        elif model_type == "linear":
            if "predicted_da_linear" in results_df.columns:
                results_df["predicted_da"] = results_df["predicted_da_linear"]
        elif model_type == "rf":
            if "predicted_da_rf" in results_df.columns:
                results_df["predicted_da"] = results_df["predicted_da_rf"]
        elif model_type in ("ensemble",):
            if "ensemble_prediction" in results_df.columns:
                results_df["predicted_da"] = results_df["ensemble_prediction"]
        # else: predicted_da stays as XGBoost prediction (default from _run_single_raw_validation)

        # For predicted_category, use the final predicted_da
        pred_col = "predicted_da"
        if model_type == "logistic" and "predicted_category_logistic" in results_df.columns:
            results_df["predicted_category"] = results_df["predicted_category_logistic"]
        elif pred_col in results_df.columns:
            results_df["predicted_category"] = (
                self.classification_adapter.threshold_classify_series(
                    results_df[pred_col]
                )
            )

        # Sort and deduplicate (same date+site can appear multiple times in test_samples)
        n_before_dedup = len(results_df)
        results_df = results_df.sort_values(["date", "site"]).drop_duplicates(
            ["date", "site"]
        )
        n_after_dedup = len(results_df)
        n_dedup_dropped = n_before_dedup - n_after_dedup
        if n_dedup_dropped > 0:
            logger.info(
                "Retrospective: dropped %d duplicate (date, site) rows -> %d unique predictions",
                n_dedup_dropped, n_after_dedup,
            )
        # One-line summary: why run count (n_processed) can exceed saved count (n_after_dedup)
        if n_processed != n_after_dedup:
            logger.info(
                "Retrospective: %d samples run -> %d saved (%d failed, %d duplicate date+site)",
                n_processed, n_after_dedup, n_failed, n_dedup_dropped,
            )

        self.results_df = results_df
        self._display_evaluation_metrics(task)

        return results_df

    def _run_single_retrospective(
        self,
        raw_measurement: dict,
        feature_frame: pd.DataFrame,
        base_params: dict,
    ) -> Optional[dict]:
        """
        Run a single retrospective validation point with per-anchor tuning.

        Mirrors ``run_single_raw_validation_with_tuning()`` from the source.
        """
        test_date = raw_measurement["date"]
        site = raw_measurement["site"]
        actual_da = raw_measurement["da_raw"]
        forecast_horizon = config.FORECAST_HORIZON_DAYS
        anchor_date = test_date - pd.Timedelta(days=forecast_horizon)

        use_per_site = getattr(config, "USE_PER_SITE_MODELS", True)
        calibration_frac = getattr(config, "CALIBRATION_FRACTION", 0.3)
        max_calib = getattr(config, "MAX_CALIBRATION_ROWS", 20)
        min_tuning = getattr(config, "MIN_TUNING_SAMPLES", 10)
        param_grid = getattr(config, "PARAM_GRID", [])

        # Per-site XGB param overrides
        effective_params = (
            apply_site_xgb_params(base_params, site) if use_per_site else base_params
        )

        train_data = get_site_training_frame(
            feature_frame, site, anchor_date, self.min_training_samples
        )
        if train_data is None or train_data.empty:
            return None

        # --- Per-anchor tuning ---
        calib_candidates = train_data[["date", "site", "da_raw"]].dropna().copy()

        # Site-specific param grid
        if use_per_site:
            site_grid = get_site_param_grid(site)
            if site_grid is not None:
                effective_grid = list(site_grid) + [
                    g for g in param_grid if g not in site_grid
                ]
            else:
                effective_grid = param_grid
        else:
            effective_grid = param_grid

        best_params = effective_params

        if (
            not calib_candidates.empty
            and len(calib_candidates) >= min_tuning
            and len(effective_grid) > 1
        ):
            # Sample calibration rows
            rng_seed = self.random_seed + int(test_date.value % 1_000_000)
            rng = np.random.RandomState(rng_seed)
            n_cand = len(calib_candidates)
            target_n = min(max(1, int(np.ceil(calibration_frac * n_cand))), max_calib)
            if target_n < n_cand:
                idx = rng.choice(n_cand, size=target_n, replace=False)
                calib_candidates = calib_candidates.iloc[idx]

            calib_rows = [
                {"date": row["date"], "site": row["site"], "da_raw": row["da_raw"]}
                for _, row in calib_candidates.iterrows()
            ]

            if len(calib_rows) >= min_tuning:
                best_params, _ = self._tune_xgb_params(
                    calib_rows, feature_frame, effective_params, effective_grid
                )
        elif len(effective_grid) == 1:
            best_params = {**effective_params, **effective_grid[0]}

        # --- Run the actual prediction ---
        return self._run_single_raw_validation(
            raw_measurement, feature_frame, best_params, skip_quantiles=False
        )

    def _run_single_raw_validation(
        self,
        raw_measurement: dict,
        feature_frame: pd.DataFrame,
        model_params: dict,
        skip_quantiles: bool = False,
        skip_rf: bool = False,
    ) -> Optional[dict]:
        """
        Core single-point validation logic for a single anchor/measurement.
        """
        test_date = raw_measurement["date"]
        site = raw_measurement["site"]
        actual_da = raw_measurement["da_raw"]

        forecast_horizon = config.FORECAST_HORIZON_DAYS
        anchor_date = test_date - pd.Timedelta(days=forecast_horizon)
        use_per_site = getattr(config, "USE_PER_SITE_MODELS", True)
        zero_imp = getattr(config, "ZERO_IMPORTANCE_FEATURES", [])
        clip_q = getattr(config, "PREDICTION_CLIP_Q", 0.99)

        train_data = get_site_training_frame(
            feature_frame, site, anchor_date, self.min_training_samples
        )
        if train_data is None:
            return None

        test_row = get_site_anchor_row(
            feature_frame, site, test_date, anchor_date, max_date_diff_days=28
        )
        if test_row is None:
            return None

        test_row = recompute_test_row_persistence_features(
            test_row, train_data, config.SPIKE_THRESHOLD
        )

        train_data = add_temporal_features(train_data)
        test_row = add_temporal_features(test_row)

        drop_cols = ["date", "site", "da_raw", "da"] + list(zero_imp)
        if use_per_site:
            drop_cols = compute_site_drop_cols(
                drop_cols, train_data.columns.tolist(), site
            )

        try:
            transformer, X_train = create_transformer(train_data, drop_cols)
            y_train = train_data["da_raw"].astype(float).copy()

            X_train_processed = transformer.fit_transform(X_train)

            X_test = test_row.drop(columns=drop_cols, errors="ignore")
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
            X_test_processed = transformer.transform(X_test)

            _verify_no_data_leakage(train_data, test_date, anchor_date)
        except Exception:
            return None

        # Post-processing
        def _postprocess(value: float) -> float:
            value = max(0.0, value)
            if use_per_site:
                sq, sm = get_site_clip_params(site)
                cq = sq if sq is not None else clip_q
            else:
                cq = clip_q
                sm = None
            if cq is not None:
                clip_max = float(np.quantile(train_data["da_raw"], cq))
                value = min(value, clip_max)
            if sm is not None:
                value = min(value, sm)
            return float(value)

        # XGBoost
        xgb_model = build_xgb_regressor(model_params)
        xgb_model.fit(X_train_processed, y_train)
        xgb_raw = float(xgb_model.predict(X_test_processed)[0])
        prediction = _postprocess(xgb_raw)

        # Random Forest
        rf_prediction = None
        if not skip_rf:
            try:
                rf_base = dict(getattr(config, "RF_REGRESSION_PARAMS", {}))
                if use_per_site:
                    rf_base = apply_site_rf_params(rf_base, site)
                rf_model = build_rf_regressor(rf_base)
                rf_model.fit(X_train_processed, y_train)
                rf_raw = float(rf_model.predict(X_test_processed)[0])
                rf_prediction = _postprocess(rf_raw)
            except Exception:
                rf_prediction = prediction

        # Naive
        naive_prediction = get_last_known_raw_da(
            train_data,
            anchor_date=anchor_date,
            max_age_days=getattr(config, "PERSISTENCE_MAX_DAYS", None),
        )
        if naive_prediction is None:
            return None

        # Ridge (linear competitor, only when needed for retrospective)
        linear_prediction = None
        if getattr(self, "_retro_model_type", None) == "linear":
            try:
                linear_model = build_linear_regressor()
                linear_model.fit(X_train_processed, y_train)
                linear_raw = float(linear_model.predict(X_test_processed)[0])
                linear_prediction = _postprocess(linear_raw)
            except Exception:
                linear_prediction = None

        # Logistic classification (only when needed for retrospective)
        predicted_category_logistic = None
        if getattr(self, "_retro_model_type", None) == "logistic":
            try:
                y_categories = self.classification_adapter.threshold_classify_series(
                    train_data["da_raw"]
                )
                unique_cats = sorted(y_categories.unique())
                if len(unique_cats) >= 2:
                    logistic_model = build_logistic_classifier()
                    logistic_model.fit(X_train_processed, y_categories)
                    predicted_category_logistic = int(
                        logistic_model.predict(X_test_processed)[0]
                    )
            except Exception:
                predicted_category_logistic = None

        # Quantile intervals
        quantile_predictions = {}
        enable_qi = getattr(config, "ENABLE_QUANTILE_INTERVALS", True)
        if enable_qi and not skip_quantiles:
            try:
                for q in (0.1, 0.5, 0.9):
                    q_params = {
                        **model_params,
                        "objective": "reg:quantile",
                        "quantile_alpha": q,
                    }
                    q_model = build_xgb_regressor(q_params)
                    q_model.fit(X_train_processed, y_train)
                    q_pred = float(q_model.predict(X_test_processed)[0])
                    quantile_predictions[f"predicted_p{int(q * 100)}"] = _postprocess(
                        q_pred
                    )
            except Exception:
                quantile_predictions = {}

        # Feature importance
        feature_importance = {}
        if hasattr(xgb_model, "feature_importances_"):
            try:
                feat_names = X_train.columns.tolist()
                feature_importance = dict(
                    zip(feat_names, xgb_model.feature_importances_)
                )
            except Exception:
                pass

        result = {
            "test_date": test_date,
            "anchor_date": anchor_date,
            "processed_test_date": test_row["date"].iloc[0],
            "site": site,
            "actual_da_raw": actual_da,
            "predicted_da": prediction,
            "predicted_da_rf": rf_prediction if rf_prediction is not None else prediction,
            "naive_prediction": naive_prediction,
            "predicted_da_linear": linear_prediction,
            "predicted_category_logistic": predicted_category_logistic,
            "training_samples": len(train_data),
            "days_ahead": (test_date - anchor_date).days,
            "date_diff_to_processed": int(
                abs((test_row["date"].iloc[0] - test_date).days)
            ),
            "feature_importance": feature_importance,
        }

        return {**result, **quantile_predictions}

    def _tune_xgb_params(
        self,
        calib_rows: list[dict],
        feature_frame: pd.DataFrame,
        base_params: dict,
        grid: list[dict],
    ) -> tuple[dict, float]:
        """Tune XGB hyperparameters on calibration rows."""
        best_params = base_params
        best_r2 = float("-inf")

        for override in grid:
            params = {**base_params, **override}
            results = [
                self._run_single_raw_validation(
                    row, feature_frame, params, skip_quantiles=True, skip_rf=True
                )
                for row in calib_rows
            ]
            results = [r for r in results if r is not None]
            if not results:
                continue
            df = pd.DataFrame(results)
            try:
                r2 = r2_score(df["actual_da_raw"].values, df["predicted_da"].values)
            except Exception:
                continue
            if r2 > best_r2:
                best_r2 = r2
                best_params = params

        return best_params, best_r2

    # ------------------------------------------------------------------
    # Metrics display (backward compat)
    # ------------------------------------------------------------------

    def _display_evaluation_metrics(self, task: str):
        """Display evaluation metrics to log."""
        if self.results_df is None or self.results_df.empty:
            return

        logger.info("Processed %d retrospective forecasts", len(self.results_df))

        if task in ("regression", "both"):
            valid = self.results_df.dropna(subset=["actual_da", "predicted_da"])
            if not valid.empty:
                r2 = r2_score(valid["actual_da"], valid["predicted_da"])
                mae = mean_absolute_error(valid["actual_da"], valid["predicted_da"])
                yt = (valid["actual_da"] > config.SPIKE_THRESHOLD).astype(int)
                yp = (valid["predicted_da"] > config.SPIKE_THRESHOLD).astype(int)
                f1_val = f1_score(yt, yp, zero_division=0)
                logger.info(
                    "Regression — R2: %.4f  MAE: %.4f  Spike F1: %.4f",
                    r2,
                    mae,
                    f1_val,
                )

        if task in ("classification", "both"):
            valid = self.results_df.dropna(
                subset=["actual_category", "predicted_category"]
            )
            if not valid.empty:
                acc = accuracy_score(valid["actual_category"], valid["predicted_category"])
                logger.info("Classification — Accuracy: %.4f", acc)

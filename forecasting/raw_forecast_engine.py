"""
Raw-Data Forecast Engine
========================

Drop-in replacement for the original ``ForecastEngine`` that uses the
raw-data ensemble pipeline (XGBoost + Random Forest two-model ML ensemble, with naïve persistence as an external standalone baseline)
with per-site hyperparameter tuning, observation-order lag features,
leak-free anchor-date environmental features, and quantile prediction
intervals.

Provides the **exact same method signatures** as the old engine so the
FastAPI backend (``api.py``) works without interface changes.
"""

from __future__ import annotations

import functools
import random
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
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
from .validation import validate_system_startup
from .logging_config import get_logger
from .metrics import detection_scores, exceeds

logger = get_logger(__name__)


def _quiet_warnings(fn):
    """Silence third-party (sklearn/xgboost/pandas) warnings during model fitting.

    Scoped per call so importing the engine no longer disables warnings for the
    whole process (tests, API, scripts).
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)
    return wrapper


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
            1, int(config.MIN_TRAINING_SAMPLES)
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
    # Pooled cross-site training data for spike classifier
    # ------------------------------------------------------------------

    def _build_pooled_spike_data(
        self,
        feature_frame: pd.DataFrame,
        anchor_date: pd.Timestamp,
    ):
        """Build pooled ALL-SITE training data for the spike binary classifier.

        Uses data from every site up to *anchor_date* (temporal integrity),
        keeping only rows with real ``da_raw`` measurements.  No per-site
        feature selection is applied — the spike classifier gets a uniform
        feature set across all sites.

        Returns ``(X_pooled, y_da_raw, transformer, columns)``: the fitted
        pooled transformer and its input columns must also be used to encode
        the test row, so it is scored on the scale the classifier was trained
        on. Returns four Nones if there is too little data.
        """
        anchor_date = pd.Timestamp(anchor_date)
        all_train = feature_frame[feature_frame["date"] <= anchor_date].copy()

        # Only real measurements — can't define "spike" from interpolated DA
        all_train = all_train[all_train["da_raw"].notna()].copy()
        if len(all_train) < 30:
            return None, None, None, None

        all_train = add_temporal_features(all_train)

        y_raw = all_train["da_raw"].astype(float).copy()

        # Standard drops: meta cols + zero-importance features
        # NO per-site drops — spike classifier uses uniform features
        zero_imp = config.ZERO_IMPORTANCE_FEATURES
        drop_cols = [
            "date", "site", "da_raw", "da", "_is_interpolated",
        ] + list(zero_imp)

        transformer, X_pooled = create_transformer(all_train, drop_cols)
        columns = list(X_pooled.columns)
        X_pooled = transformer.fit_transform(X_pooled)

        return X_pooled, y_raw, transformer, columns

    # ------------------------------------------------------------------
    # Single forecast (realtime)
    # ------------------------------------------------------------------

    @_quiet_warnings
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

        Point predictions come from ``_run_single_raw_validation``, the same
        models the retrospective evaluation scores, so the dashboard shows the
        evaluated system. This method adds what only the UI needs: the selected
        model's prediction, prediction intervals and risk categories.

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
        feature_frame = self._load_feature_frame()

        xgb_params = dict(config.XGB_REGRESSION_PARAMS)
        if config.USE_PER_SITE_MODELS:
            xgb_params = apply_site_xgb_params(xgb_params, site)

        core = self._run_single_raw_validation(
            {"date": forecast_date, "site": site, "da_raw": np.nan},
            feature_frame,
            xgb_params,
            skip_quantiles=True,
            return_internals=True,
        )
        if core is None:
            logger.warning("Insufficient data for a %s forecast on %s", site, forecast_date.date())
            return None
        fitted = core["_internals"]

        xgb_prediction = core["predicted_da"]
        rf_prediction = core["predicted_da_rf"]
        naive_prediction = core["naive_prediction"]
        linear_prediction = core["predicted_da_linear"]

        # --- Ensemble blend ---
        w_xgb, w_rf, w_naive = self._ensemble_weights(site)
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

        # Build result dict matching the API contract
        result = {
            "forecast_date": forecast_date,
            "anchor_date": core["anchor_date"],
            "site": site,
            "task": task,
            "model_type": model_type,
            "training_samples": core["training_samples"],
            "predicted_da": float(primary_prediction),
            "feature_importance": core["feature_importance"],
            # Ensemble breakdown
            "naive_prediction": float(naive_prediction),
            "xgb_prediction": float(xgb_prediction),
            "rf_prediction": float(rf_prediction),
            "ensemble_prediction": float(ensemble_prediction),
            "ensemble_weights": [float(w_xgb), float(w_rf), float(w_naive)],
        }

        # --- Quantile / bootstrap confidence intervals ---
        if task == "regression":
            quantiles = self._compute_confidence_intervals(
                fitted["X_train_processed"], fitted["y_train"], fitted["X_test_processed"],
                fitted["xgb_params"], fitted["postprocess"], model_type, site,
                naive_prediction, fitted["rf_params"], (w_xgb, w_rf, w_naive),
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
                    fitted["train_data"]["da_raw"]
                )
                unique_cats = sorted(y_categories.unique())
                if len(unique_cats) >= 2:
                    try:
                        logistic_model = build_logistic_classifier()
                        logistic_model.fit(fitted["X_train_processed"], y_categories)
                        result["predicted_category"] = int(
                            logistic_model.predict(fitted["X_test_processed"])[0]
                        )
                        if hasattr(logistic_model, "predict_proba"):
                            probs = logistic_model.predict_proba(fitted["X_test_processed"])[0]
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
                            fitted["X_train_processed"],
                            fitted["train_data"]["da_raw"],
                            site=site,
                        )
                    )
                    if classifier_result is not None:
                        full_cls = self.classification_adapter.classify_prediction(
                            primary_prediction, classifier_result, fitted["X_test_processed"],
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

        # --- Spike alert: the evaluated hybrid rule (pooled classifier OR ensemble >= alert) ---
        if core["spike_probability"] is not None:
            result["spike_probability"] = core["spike_probability"]
            result["spike_alert"] = bool(core["spike_alert"])
        elif core["spike_alert"]:
            result["spike_alert"] = True

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
        enable_quantile = config.ENABLE_QUANTILE_INTERVALS
        enable_bootstrap = config.ENABLE_BOOTSTRAP_INTERVALS

        if enable_quantile and model_type in ("xgboost", "xgb", "ensemble", "threshold"):
            try:
                # RF has no quantile objective: use the spread of its trees' predictions
                # (one fit serves every quantile). A single RF point would give RF-only
                # sites a zero-width interval.
                rf_tree_preds = None
                for q in (0.05, 0.50, 0.95):
                    # XGBoost quantile regression for the XGB component
                    q_params = {
                        **xgb_params,
                        "objective": "reg:quantileerror",
                        "quantile_alpha": q,
                    }
                    q_model = build_xgb_regressor(q_params)
                    q_model.fit(X_train_processed, y_train)
                    xgb_q_pred = postprocess_fn(float(q_model.predict(X_test_processed)[0]))

                    key = f"q{int(q * 100):02d}"
                    if model_type in ("ensemble", "threshold"):
                        # Blend XGB quantile with RF + naive for ensemble CI
                        if rf_tree_preds is None:
                            rf_model = build_rf_regressor(rf_params)
                            rf_model.fit(X_train_processed, y_train)
                            x_test = np.asarray(X_test_processed)
                            rf_tree_preds = np.array(
                                [tree.predict(x_test)[0] for tree in rf_model.estimators_]
                            )
                        rf_q_pred = postprocess_fn(float(np.quantile(rf_tree_preds, q)))
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
        subsample_frac = config.BOOTSTRAP_SUBSAMPLE_FRACTION
        if postprocess_fn is None:
            postprocess_fn = lambda val: max(0.0, float(val))

        rf_params = rf_params or dict(config.RF_REGRESSION_PARAMS)
        if ensemble_weights is None:
            # naive_weight must always be 0.0 — naive persistence is an external baseline,
            # never blended into the ensemble. Default splits evenly between XGB and RF.
            ensemble_weights = (0.50, 0.50, 0.00)
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
        percentiles = config.CONFIDENCE_PERCENTILES
        return {
            "q05": float(np.percentile(predictions, percentiles[0])),
            "q50": float(np.percentile(predictions, percentiles[1])),
            "q95": float(np.percentile(predictions, percentiles[2])),
            "bootstrap_predictions": predictions.tolist(),
        }

    # ------------------------------------------------------------------
    # Retrospective evaluation
    # ------------------------------------------------------------------

    @_quiet_warnings
    def run_retrospective_evaluation(
        self,
        task: str = "regression",
        model_type: str = "ensemble",
        max_test_samples: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Run leak-free retrospective evaluation on raw DA measurements.

        Returns DataFrame with columns matching the API's expected
        canonical keys: ``date``, ``site``, ``anchor_date``, ``actual_da``,
        ``predicted_da``, ``actual_category``, ``predicted_category``.

        If *max_test_samples* is set, the per-site draw is subsampled to at
        most that many rows (after concatenation), for quick smoke tests.
        """
        logger.info("Running LEAK-FREE %s evaluation with %s", task, model_type)
        self._retro_model_type = model_type

        feature_frame = self._load_feature_frame()
        raw_data = self._raw_data_cache

        min_test_ts = pd.Timestamp(config.MIN_TEST_DATE)
        min_training = self.min_training_samples
        forecast_horizon = config.FORECAST_HORIZON_DAYS
        history_frac = config.HISTORY_REQUIREMENT_FRACTION

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
            test_frac = config.TEST_SAMPLE_FRACTION
            target = min(int(np.ceil(test_frac * total_site)), n_candidates)
            if target <= 0:
                continue
            idx = rng.choice(n_candidates, size=target, replace=False)
            sampled_rows.append(site_df.iloc[idx])

        if not sampled_rows:
            logger.warning("Per-site sampling produced no samples")
            return None

        test_samples = pd.concat(sampled_rows, ignore_index=True)
        if max_test_samples is not None and len(test_samples) > max_test_samples:
            test_samples = test_samples.sample(
                n=max_test_samples,
                random_state=self.random_seed,
            ).reset_index(drop=True)
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
        enable_parallel = config.ENABLE_PARALLEL
        n_jobs = config.N_JOBS

        parallel_backend = config.PARALLEL_BACKEND
        if enable_parallel:
            results = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
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
        use_per_site = config.USE_PER_SITE_MODELS
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
                # Fallback: equal XGB/RF split, no naive blending (naive is external baseline)
                w_xgb, w_rf, w_naive = 0.50, 0.50, 0.00
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

        # Sort and deduplicate (defensive: raw measurements are already one row per site-day)
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

    @_quiet_warnings
    def _run_single_retrospective(
        self,
        raw_measurement: dict,
        feature_frame: pd.DataFrame,
        base_params: dict,
        return_internals: bool = False,
    ) -> Optional[dict]:
        """
        Run a single retrospective validation point with per-anchor tuning.

        Mirrors ``run_single_raw_validation_with_tuning()`` from the source.
        ``return_internals`` is passed through to ``_run_single_raw_validation``.
        """
        test_date = raw_measurement["date"]
        site = raw_measurement["site"]
        forecast_horizon = config.FORECAST_HORIZON_DAYS
        anchor_date = test_date - pd.Timedelta(days=forecast_horizon)

        use_per_site = config.USE_PER_SITE_MODELS
        calibration_frac = config.CALIBRATION_FRACTION
        max_calib = config.MAX_CALIBRATION_ROWS
        min_tuning = config.MIN_TUNING_SAMPLES
        param_grid = config.PARAM_GRID

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

        # Skip tuning when the full training set is too small — every row matters
        # more than marginal param selection between near-identical grid entries.
        min_train_for_tuning = config.MIN_TRAINING_FOR_TUNING

        if (
            config.PER_ANCHOR_TUNING
            and not calib_candidates.empty
            and len(calib_candidates) >= min_tuning
            and len(effective_grid) > 1
            and len(train_data) >= min_train_for_tuning
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
        # skip_quantiles=True: retrospective evaluation does not use quantile
        # intervals, so avoid the 3 extra XGB quantile model trainings per point.
        return self._run_single_raw_validation(
            raw_measurement, feature_frame, best_params, skip_quantiles=True,
            return_internals=return_internals,
        )

    def _run_single_raw_validation(
        self,
        raw_measurement: dict,
        feature_frame: pd.DataFrame,
        model_params: dict,
        skip_quantiles: bool = False,
        skip_rf: bool = False,
        return_internals: bool = False,
    ) -> Optional[dict]:
        """
        Core single-point forecast: the models every reported number is scored on.

        The retrospective evaluation calls this per test point and the dashboard's
        single forecasts call it too (``return_internals=True`` adds the fitted
        inputs under ``"_internals"`` so intervals and categories can be built on
        exactly the same data).
        """
        test_date = raw_measurement["date"]
        site = raw_measurement["site"]
        actual_da = raw_measurement["da_raw"]

        forecast_horizon = config.FORECAST_HORIZON_DAYS
        anchor_date = test_date - pd.Timedelta(days=forecast_horizon)
        use_per_site = config.USE_PER_SITE_MODELS
        zero_imp = config.ZERO_IMPORTANCE_FEATURES
        clip_q = config.PREDICTION_CLIP_Q

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

        # Recompute persistence features from real observations only
        if "_is_interpolated" in train_data.columns:
            real_train = train_data[~train_data["_is_interpolated"]]
        else:
            real_train = train_data
        test_row = recompute_test_row_persistence_features(
            test_row, real_train, config.SPIKE_THRESHOLD
        )

        train_data = add_temporal_features(train_data)
        test_row = add_temporal_features(test_row)

        drop_cols = ["date", "site", "da_raw", "da", "_is_interpolated"] + list(zero_imp)
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

        # XGBoost — optionally add monotonic constraints on persistence features
        xgb_params_final = dict(model_params)
        if config.USE_MONOTONIC_CONSTRAINTS:
            constraint_map = config.MONOTONIC_FEATURE_CONSTRAINTS
            if constraint_map and hasattr(X_train_processed, "columns"):
                cols = list(X_train_processed.columns)
                mono_tuple = tuple(constraint_map.get(c, 0) for c in cols)
                if any(v != 0 for v in mono_tuple):
                    xgb_params_final["monotone_constraints"] = mono_tuple
        xgb_model = build_xgb_regressor(xgb_params_final)
        xgb_model.fit(X_train_processed, y_train)
        xgb_raw = float(xgb_model.predict(X_test_processed)[0])
        prediction = _postprocess(xgb_raw)

        # Random Forest
        rf_prediction = None
        rf_base = dict(config.RF_REGRESSION_PARAMS)
        if use_per_site:
            rf_base = apply_site_rf_params(rf_base, site)
        if not skip_rf:
            try:
                rf_model = build_rf_regressor(rf_base)
                rf_model.fit(X_train_processed, y_train)
                rf_raw = float(rf_model.predict(X_test_processed)[0])
                rf_prediction = _postprocess(rf_raw)
            except Exception as e:
                logger.warning("RF training failed for %s on %s: %s — using XGB prediction only", site, test_date, e)
                rf_prediction = prediction

        # Naive (always use real observations only)
        naive_prediction = get_last_known_raw_da(
            real_train,
            anchor_date=anchor_date,
            max_age_days=config.PERSISTENCE_MAX_DAYS,
        )
        if naive_prediction is None:
            return None

        # Ridge (always trained as scientific baseline for "ML earns its keep" comparison)
        # Negligible cost — single matrix solve. Ensemble stays ML-only regardless.
        linear_prediction = None
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
        enable_qi = config.ENABLE_QUANTILE_INTERVALS
        if enable_qi and not skip_quantiles:
            try:
                for q in (0.1, 0.5, 0.9):
                    q_params = {
                        **model_params,
                        "objective": "reg:quantileerror",
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

        # Spike binary classifier (pooled cross-site training)
        spike_prob = None
        spike_alert = None
        if config.SPIKE_CLASSIFIER_ENABLED:
            try:
                X_pooled, y_pooled, pooled_transformer, pooled_columns = (
                    self._build_pooled_spike_data(feature_frame, anchor_date)
                )
                if X_pooled is not None:
                    spike_result = (
                        self.classification_adapter.train_spike_binary_classifier(
                            X_pooled,
                            y_pooled,
                            spike_threshold=config.SPIKE_THRESHOLD,
                            site=site,
                        )
                    )
                    if spike_result is not None:
                        # Encode the test row with the pooled transformer the classifier
                        # was trained on (not the per-site one, whose scaling and feature
                        # subset differ); missing features get the pooled median.
                        X_test_spike = pooled_transformer.transform(
                            test_row.reindex(columns=pooled_columns)
                        )
                        spike_prob = (
                            self.classification_adapter.predict_spike_probability(
                                spike_result, X_test_spike,
                            )
                        )
                        classifier_fires = spike_prob >= config.SPIKE_ALERT_PROB_THRESHOLD
                        w_xgb, w_rf, _ = self._ensemble_weights(site)
                        rf_val = rf_prediction if rf_prediction is not None else prediction
                        ensemble_pred = w_xgb * prediction + w_rf * rf_val
                        regression_fires = ensemble_pred >= config.SPIKE_REGRESSION_ALERT_THRESHOLD
                        spike_alert = classifier_fires or regression_fires
            except Exception as exc:
                logger.debug("Spike classifier failed in validation: %s", exc)

        # Regression-only fallback when classifier is unavailable
        if spike_alert is None:
            w_xgb, w_rf, _ = self._ensemble_weights(site)
            rf_val = rf_prediction if rf_prediction is not None else prediction
            ensemble_pred = w_xgb * prediction + w_rf * rf_val
            if ensemble_pred >= config.SPIKE_REGRESSION_ALERT_THRESHOLD:
                spike_alert = True

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
            "spike_probability": spike_prob,
            "spike_alert": spike_alert,
        }

        if return_internals:
            result["_internals"] = {
                "train_data": train_data,
                "X_train": X_train,
                "X_train_processed": X_train_processed,
                "X_test_processed": X_test_processed,
                "y_train": y_train,
                "postprocess": _postprocess,
                "xgb_params": xgb_params_final,
                "rf_params": rf_base,
            }
        return {**result, **quantile_predictions}

    @staticmethod
    def _ensemble_weights(site: str) -> tuple[float, float, float]:
        """(w_xgb, w_rf, w_naive): per-site weights, or an equal XGB/RF split when
        USE_PER_SITE_MODELS is off (naive persistence is an external baseline)."""
        if config.USE_PER_SITE_MODELS:
            return get_site_ensemble_weights(site)
        return 0.5, 0.5, 0.0

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
                f1_val = detection_scores(
                    exceeds(valid["actual_da"], config.SPIKE_THRESHOLD),
                    exceeds(valid["predicted_da"], config.SPIKE_REGRESSION_ALERT_THRESHOLD),
                )["f1"]
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

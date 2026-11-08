"""
Raw-Data Forecast Engine
========================

Drop-in replacement for the original ``ForecastEngine`` that uses the
raw-data ensemble pipeline (XGBoost + Random Forest two-model ML ensemble, with naïve persistence as an external standalone baseline)
with per-site hyperparameter tuning, observation-order lag features and
leak-free anchor-date environmental features. It computes no prediction
interval: the dashboard's uncertainty is the conformal error band
(``backend/band.py``).

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
    WEEK_END_OFFSET,
    week_complete_by,
)
from .raw_model_factory import (
    build_xgb_regressor,
    build_rf_regressor,
    build_linear_regressor,
)
from .per_site_models import (
    apply_site_xgb_params,
    apply_site_rf_params,
    blend,
    get_site_ensemble_weights,
    get_site_clip_params,
    compute_site_drop_cols,
)
from .feature_utils import add_temporal_features, create_transformer
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
# Helpers: supported requests; leakage verification (ported from source config.py)
# ---------------------------------------------------------------------------

# The model_type values callers pass: the API and the store builder (ensemble,
# xgboost, rf, naive, linear) and the eval scripts ("xgb", an alias for xgboost).
SUPPORTED_MODELS = ("ensemble", "xgboost", "xgb", "rf", "naive", "linear")


def _require_regression(task: str, model_type: str) -> None:
    """The engine forecasts DA levels only, with one of ``SUPPORTED_MODELS``. The
    4-category classifier and the logistic model were removed (categories come from
    the predicted level, and on the dashboard from the error band), so asking for
    them, or for any other model, fails here instead of falling through to another
    model."""
    if task != "regression":
        raise ValueError(f"Unknown task '{task}': the engine supports 'regression' only")
    if model_type == "logistic":
        raise ValueError("model_type 'logistic' was removed with the 4-category classifier; "
                         "the linear competitor is 'linear' (Ridge)")
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model_type '{model_type}': the engine supports "
                         + ", ".join(f"'{m}'" for m in SUPPORTED_MODELS) + " only")


def _verify_no_data_leakage(train_data, test_date, anchor_date):
    """Assert no training week extends past the anchor date."""
    anchor = pd.Timestamp(anchor_date)
    last_week_end = train_data["date"].max() + WEEK_END_OFFSET
    if last_week_end > anchor:
        raise AssertionError(
            f"TEMPORAL LEAK: training week ending {last_week_end.date()} "
            f"is after anchor {anchor.date()}"
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

    Replaces the original ``ForecastEngine``; its public API is
    ``generate_single_forecast`` and ``run_retrospective_evaluation``.
    """

    def __init__(self, data_file=None, validate_on_init=True):
        logger.info("Initializing RawForecastEngine")
        self.data_file = data_file or config.FINAL_OUTPUT_PATH
        self.results_df = None
        self._feature_frame_cache: Optional[pd.DataFrame] = None
        self._raw_data_cache: Optional[pd.DataFrame] = None

        if validate_on_init:
            validate_system_startup()

        self.classification_adapter = ClassificationAdapter()

        self.min_training_samples = max(
            1, int(config.MIN_TRAINING_SAMPLES)
        )
        self.random_seed = config.RANDOM_SEED

        # No engine code draws from the global RNGs (every model takes
        # config.MODEL_SEED; sampling uses its own RandomState). They stay seeded
        # so an implicit consumer a grep cannot see would still be deterministic.
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

    def _observations_up_to(self, site: str, anchor_date: pd.Timestamp) -> pd.DataFrame:
        """Real DA measurements at ``site`` dated on or before ``anchor_date``
        (one per day), sorted by date: the source of persistence features and
        the naive baseline, at daily rather than weekly resolution."""
        if self._raw_data_cache is None:
            self._raw_data_cache = load_raw_da_measurements()
        raw = self._raw_data_cache
        obs = raw[(raw["site"] == site) & (pd.to_datetime(raw["date"]) <= anchor_date)].copy()
        obs["date"] = pd.to_datetime(obs["date"])
        return obs.sort_values("date")

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
        all_train = feature_frame[week_complete_by(feature_frame["date"], anchor_date)].copy()

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

        Point predictions come from ``_point_forecast``, the same per-point
        function the retrospective evaluation scores, so the dashboard shows the
        evaluated system. This method adds what only the UI needs: the selected
        model's prediction, the ensemble breakdown, the risk category and the
        spike alert. It computes no interval (the API adds the conformal band).

        Parameters
        ----------
        data_path : str
            Path to processed parquet (used for cache key only).
        forecast_date : date-like
            Date to forecast for.
        site : str
            Monitoring site name.
        task : str
            ``"regression"``, the only task; anything else raises ``ValueError``.
        model_type : str
            One of ``SUPPORTED_MODELS``: ``"ensemble"``, ``"xgboost"`` (alias
            ``"xgb"``), ``"rf"``, ``"naive"`` or ``"linear"``; anything else
            raises ``ValueError``.

        Returns
        -------
        dict or None
        """
        _require_regression(task, model_type)
        forecast_date = pd.Timestamp(forecast_date)
        feature_frame = self._load_feature_frame()

        # No n_jobs here: single forecasts keep XGBoost's default threads.
        core = self._point_forecast(
            {"date": forecast_date, "site": site, "da_raw": np.nan},
            feature_frame,
            dict(config.XGB_REGRESSION_PARAMS),
        )
        if core is None:
            logger.warning("Insufficient data for a %s forecast on %s", site, forecast_date.date())
            return None

        xgb_prediction = core["predicted_da"]
        rf_prediction = core["predicted_da_rf"]
        naive_prediction = core["naive_prediction"]
        linear_prediction = core["predicted_da_linear"]

        # --- Ensemble blend ---
        w_xgb, w_rf, w_naive = self._ensemble_weights(site)
        ensemble_prediction = blend(
            xgb_prediction, rf_prediction, naive_prediction, (w_xgb, w_rf, w_naive)
        )

        # Choose which prediction to expose as primary (_require_regression admits
        # only SUPPORTED_MODELS)
        if model_type == "ensemble":
            primary_prediction = ensemble_prediction
        elif model_type in ("xgboost", "xgb"):
            primary_prediction = xgb_prediction
        elif model_type == "rf":
            primary_prediction = rf_prediction
        elif model_type == "naive":
            primary_prediction = naive_prediction
        else:  # "linear"
            primary_prediction = (
                linear_prediction
                if linear_prediction is not None
                else ensemble_prediction
            )

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

        # Risk category of the selected prediction (left-closed bins)
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

        Only ``task="regression"`` exists; another task, or a ``model_type``
        outside ``SUPPORTED_MODELS`` (``"logistic"`` included), raises
        ``ValueError``.
        """
        _require_regression(task, model_type)
        logger.info("Running LEAK-FREE %s evaluation with %s", task, model_type)

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
                delayed(self._point_forecast)(
                    row, feature_frame, base_params
                )
                for row in tqdm(sample_rows, desc="Retrospective", unit="sample")
            )
        else:
            results = [
                self._point_forecast(row, feature_frame, base_params)
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
        if (
            not results_df.empty
            and "predicted_da" in results_df.columns
            and "naive_prediction" in results_df.columns
        ):
            ens = []
            for _, row in results_df.iterrows():
                rf_pred = row.get("predicted_da_rf", row["predicted_da"])
                ens.append(
                    blend(
                        row["predicted_da"], rf_pred, row["naive_prediction"],
                        get_site_ensemble_weights(row["site"]),
                    )
                )
            results_df["ensemble_prediction"] = ens

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
        if pred_col in results_df.columns:
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
        self._display_evaluation_metrics()

        return results_df

    @_quiet_warnings
    def _point_forecast(
        self,
        raw_measurement: dict,
        feature_frame: pd.DataFrame,
        base_params: dict,
        return_internals: bool = False,
    ) -> Optional[dict]:
        """
        One leak-free forecast for one site and date: the function every caller uses.

        The retrospective evaluation, the Spark weight tuner (``datect_spark/retune.py``)
        and the dashboard's single forecast all come here, so the dashboard shows the
        models the reported numbers score. It merges the site's XGBoost overrides onto
        ``base_params`` when ``config.USE_PER_SITE_MODELS`` and runs
        ``_run_single_raw_validation``. XGBoost threads travel in ``base_params``: the
        retrospective and the tuner set ``n_jobs=1``, single forecasts leave the default.
        """
        params = (
            apply_site_xgb_params(base_params, raw_measurement["site"])
            if config.USE_PER_SITE_MODELS
            else base_params
        )
        return self._run_single_raw_validation(
            raw_measurement, feature_frame, params,
            return_internals=return_internals,
        )

    def _run_single_raw_validation(
        self,
        raw_measurement: dict,
        feature_frame: pd.DataFrame,
        model_params: dict,
        return_internals: bool = False,
    ) -> Optional[dict]:
        """
        Core single-point forecast: the models every reported number is scored on.

        Called only through ``_point_forecast``, which hands it the final XGBoost
        params (``return_internals=True`` adds the training frame under
        ``"_internals"``; the Spark tuner reads it).
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

        # Persistence features from real measurements dated on or before the anchor
        observations = self._observations_up_to(site, anchor_date)
        test_row = recompute_test_row_persistence_features(
            test_row, observations, config.SPIKE_THRESHOLD
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
            observations,
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
                        ensemble_pred = blend(
                            prediction, rf_prediction, 0.0, (w_xgb, w_rf, 0.0)
                        )
                        regression_fires = ensemble_pred >= config.SPIKE_REGRESSION_ALERT_THRESHOLD
                        spike_alert = classifier_fires or regression_fires
            except Exception as exc:
                logger.debug("Spike classifier failed in validation: %s", exc)

        # Regression-only fallback when classifier is unavailable
        if spike_alert is None:
            w_xgb, w_rf, _ = self._ensemble_weights(site)
            ensemble_pred = blend(prediction, rf_prediction, 0.0, (w_xgb, w_rf, 0.0))
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
            result["_internals"] = {"train_data": train_data}
        return result

    @staticmethod
    def _ensemble_weights(site: str) -> tuple[float, float, float]:
        """(w_xgb, w_rf, w_naive): per-site weights, or an equal XGB/RF split when
        USE_PER_SITE_MODELS is off (naive persistence is an external baseline)."""
        return get_site_ensemble_weights(site)

    # ------------------------------------------------------------------
    # Metrics display (backward compat)
    # ------------------------------------------------------------------

    def _display_evaluation_metrics(self):
        """Display evaluation metrics to log."""
        if self.results_df is None or self.results_df.empty:
            return

        logger.info("Processed %d retrospective forecasts", len(self.results_df))

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

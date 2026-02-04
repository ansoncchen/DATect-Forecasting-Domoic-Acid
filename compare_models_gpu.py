#!/usr/bin/env python3
"""
Compare regression performance across TFT/TCN/GPyTorch/TabNet.
Modeled after precompute_cache.py but focused on model comparison.
"""

import json
from pathlib import Path
import random

import numpy as np
import pandas as pd

import config
from forecasting.data_processor import DataProcessor
from forecasting.model_factory import ModelFactory
from forecasting.torch_forecasting_adapter import build_timeseries_dataset, prepare_timeseries_dataframe
try:
    from pytorch_forecasting import TimeSeriesDataSet
except ImportError:
    TimeSeriesDataSet = None


def _select_anchor_infos(data, n_anchors, min_test_date):
    min_target_date = pd.Timestamp(min_test_date)
    anchor_infos = []
    for site in data["site"].unique():
        site_dates = data[data["site"] == site]["date"].sort_values().unique()
        if len(site_dates) > 1:
            date_span_days = (site_dates[-1] - site_dates[0]).days
            if date_span_days >= config.FORECAST_HORIZON_DAYS * 2:
                valid_anchors = []
                for i, date in enumerate(site_dates[:-1]):
                    if date >= min_target_date:
                        future_dates = site_dates[i + 1 :]
                        valid_future = [d for d in future_dates if (d - date).days >= config.FORECAST_HORIZON_DAYS]
                        if valid_future:
                            valid_anchors.append(date)
                if valid_anchors:
                    n_sample = min(len(valid_anchors), n_anchors)
                    selected = random.sample(list(valid_anchors), n_sample)
                    anchor_infos.extend([(site, pd.Timestamp(d)) for d in selected])
    return anchor_infos


def _train_test_for_anchor(site_data, anchor_date, min_target_date):
    site_data = site_data.sort_values("date")
    target_forecast_date = anchor_date + pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
    test_mask = (site_data["date"] > anchor_date) & (site_data["date"] >= min_target_date)
    train_df = site_data[site_data["date"] <= anchor_date].copy()
    test_candidates = site_data[test_mask].copy()
    if train_df.empty or test_candidates.empty:
        return None, None
    test_candidates["date_diff"] = abs((test_candidates["date"] - target_forecast_date).dt.days)
    closest_idx = test_candidates["date_diff"].idxmin()
    test_df = test_candidates.loc[[closest_idx]].copy()
    return train_df, test_df


def _predict_tabular(model, data_processor, train_df, test_df):
    train_df = train_df.dropna(subset=["da"]).copy()
    if train_df.empty:
        return None
    drop_cols = ["date", "site", "da"]
    transformer, X_train = data_processor.create_numeric_transformer(train_df, drop_cols)
    X_test = test_df.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_train_processed = transformer.fit_transform(X_train)
    X_test_processed = transformer.transform(X_test)

    model.fit(X_train_processed.to_numpy(), train_df["da"].to_numpy())
    pred = model.predict(X_test_processed.to_numpy())[0]
    return float(max(0.0, pred))


def _predict_torch_forecasting(model, train_df, test_df):
    train_df = train_df.dropna(subset=["da"]).copy()
    if train_df.empty:
        return None

    # Build dataset from training data
    dataset, _, _ = build_timeseries_dataset(train_df)
    model.fit(dataset)

    # Build predict dataset with test row included
    if TimeSeriesDataSet is None:
        raise ImportError("pytorch-forecasting is required for TFT/TCN prediction.")
    predict_df = pd.concat([train_df, test_df], ignore_index=True)
    predict_df = prepare_timeseries_dataframe(predict_df)
    predict_dataset = TimeSeriesDataSet.from_dataset(
        dataset, predict_df, predict=True, stop_randomization=True
    )
    preds = model.predict(predict_dataset)
    if len(preds) == 0:
        return None
    return float(max(0.0, preds[-1]))


def run_model(data, model_type, n_anchors, min_test_date):
    data_processor = DataProcessor()
    model_factory = ModelFactory()
    anchor_infos = _select_anchor_infos(data, n_anchors, min_test_date)

    results = []
    min_target_date = pd.Timestamp(min_test_date)

    for site, anchor_date in anchor_infos:
        site_data = data[data["site"] == site].copy()
        train_df, test_df = _train_test_for_anchor(site_data, anchor_date, min_target_date)
        if train_df is None or test_df is None:
            continue

        if model_type in ["tft", "tcn"]:
            model = model_factory.get_model("regression", model_type)
            pred = _predict_torch_forecasting(model, train_df, test_df)
        else:
            model = model_factory.get_model("regression", model_type)
            pred = _predict_tabular(model, data_processor, train_df, test_df)

        if pred is None:
            continue

        result = {
            "date": test_df["date"].iloc[0],
            "site": site,
            "anchor_date": anchor_date,
            "actual_da": float(test_df["da"].iloc[0]) if pd.notnull(test_df["da"].iloc[0]) else None,
            "predicted_da": pred,
        }
        results.append(result)

    return pd.DataFrame(results)


def _compute_summary(df):
    from sklearn.metrics import r2_score, mean_absolute_error, f1_score

    valid = df.dropna(subset=["actual_da", "predicted_da"])
    if valid.empty:
        return {}
    y_true = valid["actual_da"].values
    y_pred = valid["predicted_da"].values
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    thr = getattr(config, "SPIKE_THRESHOLD", 20.0)
    f1 = float(f1_score((y_true > thr).astype(int), (y_pred > thr).astype(int), zero_division=0))
    return {"r2_score": r2, "mae": mae, "f1_score": f1, "n": int(len(valid))}


def main():
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("Starting model comparison")
    print("=========================")
    print(f"N_RANDOM_ANCHORS: {config.N_RANDOM_ANCHORS}")
    print(f"FORECAST_HORIZON_DAYS: {config.FORECAST_HORIZON_DAYS}")

    data_processor = DataProcessor()
    data = data_processor.load_and_prepare_base_data(config.FINAL_OUTPUT_PATH)

    cache_dir = Path("cache") / "retrospective"
    cache_dir.mkdir(parents=True, exist_ok=True)

    models = ["tft", "tcn", "gpytorch", "tabnet"]
    for model_type in models:
        print(f"Running regression for {model_type}...")
        df = run_model(data, model_type, config.N_RANDOM_ANCHORS, "2008-01-01")
        if df is None or df.empty:
            print(f"  No results for {model_type}")
            continue

        out_base = cache_dir / f"regression_{model_type}"
        df.to_parquet(f"{out_base}.parquet", index=False)
        df_json = df.copy()
        df_json["date"] = df_json["date"].astype(str)
        df_json["anchor_date"] = df_json["anchor_date"].astype(str)
        with open(f"{out_base}.json", "w") as f:
            json.dump(df_json.to_dict(orient="records"), f, indent=2)

        summary = _compute_summary(df)
        print(f"  Saved {len(df)} predictions")
        if summary:
            print(f"  Metrics: R²={summary['r2_score']:.4f}, MAE={summary['mae']:.2f}, F1={summary['f1_score']:.4f}")


if __name__ == "__main__":
    main()

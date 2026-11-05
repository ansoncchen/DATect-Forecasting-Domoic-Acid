"""
Re-tune per-site ensemble weights on Spark, every run tracked in MLflow.

The ensemble is ``w_xgb * XGBoost + (1 - w_xgb) * Random Forest``, with no clip
(removed in #18). The weights are post-processing: a test point's XGBoost and Random
Forest models are the same whatever the weights are. So instead of re-running the
whole retrospective for every weight, this:

1. ``predict --window tuning|holdout``: one Spark task per real DA measurement in
   the window. Each task trains that point's models exactly as the retrospective
   evaluation does (RawForecastEngine._run_single_retrospective) and returns the
   raw XGBoost and Random Forest predictions. Rows go to Delta.
2. ``score``: applies the weight grid (w_xgb in 0, 0.25, 0.5, 0.75, 1) to the
   tuning rows in Spark SQL and scores each (site, weight) on yearly folds
   2015-2021, all before the holdout. Each (site, weight) is an MLflow run; the
   winner maximizes mean R2 - 0.5 * std R2 over folds.
3. ``holdout``: scores the proposed and the current weights once on the 2022-2023
   holdout rows. Nothing is chosen there, and nothing is adopted: the proposal is
   written to proposed_overrides.json for a separate decision.

``predict`` also checks parity: the current weights applied to the raw predictions
must equal the committed chronological predictions.

This is the project's only weight tuner. The weights in
config/tuned_hyperparameters.json predate it: scripts/eval/grid_search_weights_clip.py
chose them jointly with a (since removed) clip on 3-fold CV over 2019-2021
(results committed in f3bc5935; the script was removed after 9c435f59). Its folds
differ from these, so re-running this tuner can propose different weights.

Usage (Hyak, inside an allocation; see datect_spark/retune.sbatch):
    python -m datect_spark.retune predict --window tuning
    python -m datect_spark.retune predict --window holdout
    python -m datect_spark.retune score
    python -m datect_spark.retune holdout
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time

import numpy as np
import pandas as pd

from datect_spark.session import get_mlflow, get_spark, git_sha, lake_path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(REPO_ROOT, "results", "tuning_provenance", "retune_weights")
EXPERIMENT = "datect-retune-weights"

W_XGB_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

PRED_SCHEMA = (
    "site string, date string, actual_da double, xgb_raw double, rf_raw double, "
    "n_train int, task_seconds double, host string, ok boolean"
)
# First tuning fold; the last is the year before the holdout (eval_windows.validation_end)
TUNING_START = "2015-01-01"

CHRONO = {
    "validation": "results/chronological/chronological_regression_ensemble_20190101_20220101.parquet",
    "holdout": "results/chronological/chronological_regression_ensemble_20220101_20240101.parquet",
}


def _window_bounds(window: str) -> tuple[str, str]:
    from forecasting.tuned_config import get_eval_windows

    w = get_eval_windows()
    if window == "tuning":
        return (TUNING_START, w["validation_end"])
    return (w[f"{window}_start"], w[f"{window}_end"])


# ── 1. Parallel training ─────────────────────────────────────────────────────

_ENGINE = None


def _engine():
    """One engine and feature frame per Python worker (workers are reused)."""
    global _ENGINE
    if _ENGINE is None:
        os.chdir(REPO_ROOT)  # config paths are relative to the repo root
        import config
        from forecasting.raw_forecast_engine import RawForecastEngine

        if config.CLIP_Q_OVERRIDE != "none":
            raise RuntimeError("workers must run with DATECT_CLIP_Q_OVERRIDE=none")
        engine = RawForecastEngine(validate_on_init=False)
        base = dict(config.XGB_REGRESSION_PARAMS)
        base["n_jobs"] = 1
        _ENGINE = (engine, engine._load_feature_frame(), base)
    return _ENGINE


def _predict_partition(batches):
    engine, feature_frame, base = _engine()
    host = socket.gethostname()
    for pdf in batches:
        rows = []
        for r in pdf.itertuples(index=False):
            t0 = time.time()
            out = engine._run_single_retrospective(
                {"date": pd.Timestamp(r.date), "site": r.site, "da_raw": r.actual_da},
                feature_frame, base, return_internals=True,
            )
            row = {"site": r.site, "date": r.date, "actual_da": r.actual_da,
                   "host": host, "ok": out is not None, "n_train": 0}
            if out is not None:
                row.update(xgb_raw=out["predicted_da"], rf_raw=out["predicted_da_rf"],
                           n_train=len(out["_internals"]["train_data"]))
            row["task_seconds"] = time.time() - t0
            rows.append(row)
        yield pd.DataFrame(rows).reindex(columns=[c.split()[0] for c in PRED_SCHEMA.split(", ")])


def predict(window: str, limit: int | None = None) -> None:
    from forecasting.raw_data_forecaster import load_raw_da_measurements

    start, end = _window_bounds(window)
    raw = load_raw_da_measurements()
    raw["date"] = pd.to_datetime(raw["date"])
    tasks = raw[(raw["date"] >= start) & (raw["date"] < end)].rename(columns={"da_raw": "actual_da"})
    # Latest dates have the most training rows: schedule them first
    tasks = tasks.sort_values("date", ascending=False).reset_index(drop=True)
    if limit:
        tasks = tasks.head(limit)  # smoke runs

    # Workers inherit this environment (local mode) or get it via executorEnv
    env = {"DATECT_CLIP_Q_OVERRIDE": "none",
           "PYTHONPATH": REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    os.environ.update(env)
    spark = get_spark(f"datect-retune-predict-{window}",
                      conf={f"spark.executorEnv.{k}": v for k, v in env.items()})
    sc = spark.sparkContext
    t0 = time.time()
    records = [(r.site, r.date.strftime("%Y-%m-%d"), float(r.actual_da)) for r in tasks.itertuples()]
    task_df = spark.createDataFrame(sc.parallelize(records, numSlices=len(records)),
                                    "site string, date string, actual_da double")
    preds = task_df.mapInPandas(_predict_partition, schema=PRED_SCHEMA).toPandas()
    wall = time.time() - t0

    table = lake_path("gold", f"retune_predictions_{window}")
    spark.createDataFrame(preds).write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").save(table)
    os.makedirs(OUT_DIR, exist_ok=True)
    preds.to_parquet(os.path.join(OUT_DIR, f"retune_predictions_{window}.parquet"), index=False)
    parity = _parity(preds, window)

    mlflow = get_mlflow(EXPERIMENT)
    with mlflow.start_run(run_name=f"predict-{window}") as parent:
        mlflow.log_params({"window": window, "start": start, "end": end, "git_sha": git_sha(REPO_ROOT),
                           "spark_master": sc.master, "default_parallelism": sc.defaultParallelism,
                           "delta_table": table})
        mlflow.log_metrics({"n_points": len(preds), "n_ok": int(preds["ok"].sum()),
                            "wall_seconds": wall, "task_seconds_total": float(preds["task_seconds"].sum()),
                            "hosts": preds["host"].nunique(), **parity})
        n_runs = _log_forecast_runs(parent, preds)
        mlflow.log_metric("n_child_runs", n_runs)
    summary = {"window": window, "points": len(preds), "ok": int(preds["ok"].sum()), "wall_seconds": wall,
               "task_seconds_total": float(preds["task_seconds"].sum()), "child_runs": n_runs, **parity}
    with open(os.path.join(OUT_DIR, f"predict_{window}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{window}: {summary['ok']}/{len(preds)} points in {wall:.0f}s wall "
          f"({summary['task_seconds_total']:.0f}s of task time on {preds['host'].nunique()} host(s)); "
          f"parity {parity}; {n_runs} forecast runs logged to MLflow")
    spark.stop()


def _log_forecast_runs(parent, preds: pd.DataFrame) -> int:
    """One MLflow child run per trained forecast (XGBoost + Random Forest pair)."""
    from mlflow.entities import Metric, Param
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    ok = preds[preds["ok"]].copy()
    ok["ensemble"] = _ensemble(ok, _current_weights())
    now = int(time.time() * 1000)
    for r in ok.itertuples(index=False):
        child = client.create_run(parent.info.experiment_id, run_name=f"{r.site}|{r.date}",
                                  tags={"mlflow.parentRunId": parent.info.run_id,
                                        "site": r.site, "host": r.host})
        values = {"actual_da": r.actual_da, "xgb_raw": r.xgb_raw, "rf_raw": r.rf_raw,
                  "ensemble_current": r.ensemble, "abs_error_current": abs(r.actual_da - r.ensemble),
                  "task_seconds": r.task_seconds}
        client.log_batch(child.info.run_id,
                         metrics=[Metric(k, float(v), now, 0) for k, v in values.items()],
                         params=[Param("site", r.site), Param("test_date", r.date),
                                 Param("n_train", str(r.n_train))])
        client.set_terminated(child.info.run_id)
    return len(ok)


# ── Weights applied to raw predictions (pandas, for parity/holdout) ──────────

def _current_weights() -> dict[str, float]:
    """Per-site w_xgb from the tuned JSON (w_rf = 1 - w_xgb). A configured clip would
    make the ensemble more than a blend of the raw predictions: not supported."""
    from forecasting.tuned_config import get_global, get_per_site

    if get_global().get("prediction_clip_q_default") is not None:
        raise RuntimeError("a default clip is configured; this tuner assumes none")
    out = {}
    for site, c in get_per_site().items():
        if c.get("prediction_clip_q") is not None or c.get("prediction_clip_max") is not None:
            raise RuntimeError(f"{site} has a clip configured; this tuner assumes none")
        w_xgb, w_rf, w_naive = c["ensemble_weights"]
        if w_naive != 0 or abs(w_xgb + w_rf - 1.0) > 1e-12:
            raise RuntimeError(f"{site} weights {c['ensemble_weights']} are not an XGB/RF blend")
        out[site] = float(w_xgb)
    return out


def _ensemble(preds: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    w = preds["site"].map(weights)
    return w * preds["xgb_raw"] + (1 - w) * preds["rf_raw"]


def _parity(preds: pd.DataFrame, window: str) -> dict:
    """Current weights applied to raw predictions vs the committed chronological run."""
    if window == "tuning":  # parity is checked on the part with a committed run
        start = pd.Timestamp(_window_bounds("validation")[0])
        return _parity(preds[pd.to_datetime(preds["date"]) >= start], "validation")
    ref = pd.read_parquet(os.path.join(REPO_ROOT, CHRONO[window]))
    ref["date"] = pd.to_datetime(ref["date"])
    ok = preds[preds["ok"]].copy()
    ok["date"] = pd.to_datetime(ok["date"])
    ok["rebuilt"] = _ensemble(ok, _current_weights())
    m = ok.merge(ref[["site", "date", "predicted_da"]], on=["site", "date"], how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    diff = (both["rebuilt"] - both["predicted_da"]).abs()
    # pandas' max skips NaN: a NaN on either side must fail parity, not vanish
    max_diff = float(diff.max()) if diff.notna().all() else float("nan")
    return {"parity_rows_matched": len(both), "parity_rows_unmatched": int((m["_merge"] != "both").sum()),
            "parity_max_abs_diff": max_diff}


def _nan_to_none(obj):
    """Strict JSON: an undefined R2 (a site with no variance) is null, not NaN."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    return None if isinstance(obj, float) and obj != obj else obj


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    sst = float(((y - y.mean()) ** 2).sum())
    return float("nan") if sst == 0 else 1.0 - float(((y - p) ** 2).sum()) / sst


# ── 2. Grid scoring in Spark SQL, one MLflow run per (site, weight) ─────────

def score() -> None:
    from pyspark.sql import functions as F

    spark = get_spark("datect-retune-score")
    preds = spark.read.format("delta").load(lake_path("gold", "retune_predictions_tuning")).where("ok")
    grid = spark.createDataFrame([(w,) for w in W_XGB_GRID], "w_xgb double")
    scored = (preds.crossJoin(grid)
              .withColumn("pred", F.col("w_xgb") * F.col("xgb_raw") + (1 - F.col("w_xgb")) * F.col("rf_raw"))
              .withColumn("fold", F.year("date"))
              .groupBy("site", "w_xgb", "fold")
              .agg(F.count("*").alias("n"),
                   F.sum((F.col("actual_da") - F.col("pred")) ** 2).alias("sse"),
                   F.sum("actual_da").alias("sy"),
                   F.sum(F.col("actual_da") ** 2).alias("syy"),
                   F.avg(F.abs(F.col("actual_da") - F.col("pred"))).alias("mae"))
              .withColumn("sst", F.col("syy") - F.col("sy") ** 2 / F.col("n"))
              .withColumn("r2", F.when((F.col("n") >= 5) & (F.col("sst") > 0),
                                       1 - F.col("sse") / F.col("sst"))))
    scored.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
        .save(lake_path("gold", "retune_fold_scores"))
    folds = scored.toPandas()
    spark.stop()

    # mean - 0.5 * std over the folds that could be scored
    agg = (folds.dropna(subset=["r2"]).groupby(["site", "w_xgb"])
           .agg(mean_r2=("r2", "mean"), std_r2=("r2", lambda s: float(np.std(s))),
                n_folds=("r2", "size"), n=("n", "sum")).reset_index())
    agg["score"] = agg["mean_r2"] - 0.5 * agg["std_r2"]
    # Spark sums in any order, so equal weights' scores can differ at ~1e-16;
    # ties go to more XGBoost weight
    agg["score_key"] = agg["score"].round(10)
    ranked = agg.sort_values(["site", "score_key", "w_xgb"], ascending=[True, False, False])
    best = ranked.groupby("site").head(1)
    agg["selected"] = agg.index.isin(best.index)
    winners = best.set_index("site")
    proposed = {site: {"ensemble_weights": [float(r.w_xgb), float(1 - r.w_xgb), 0.0]}
                for site, r in winners.iterrows()}

    os.makedirs(OUT_DIR, exist_ok=True)
    agg.drop(columns=["score_key"]).to_csv(os.path.join(OUT_DIR, "weight_scores.csv"), index=False)
    with open(os.path.join(OUT_DIR, "proposed_overrides.json"), "w") as f:
        json.dump(proposed, f, indent=2)

    mlflow = get_mlflow(EXPERIMENT)
    from mlflow.entities import Metric, Param
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    with mlflow.start_run(run_name="score-tuning-folds") as parent:
        mlflow.log_params({"git_sha": git_sha(REPO_ROOT), "objective": "mean_r2 - 0.5*std_r2",
                           "folds": "2015-2021 yearly", "weights": ",".join(map(str, W_XGB_GRID))})
        now = int(time.time() * 1000)
        for _, r in agg.iterrows():
            child = client.create_run(parent.info.experiment_id, run_name=f"{r.site}|w{r.w_xgb}",
                                      tags={"mlflow.parentRunId": parent.info.run_id, "site": r.site})
            fold_rows = folds[(folds["site"] == r.site) & (folds["w_xgb"] == r.w_xgb)]
            metrics = [Metric(k, float(v), now, 0) for k, v in
                       (("mean_r2", r.mean_r2), ("std_r2", r.std_r2), ("score", r.score), ("n", r.n))]
            metrics += [Metric(f"r2_{int(fr.fold)}", float(fr.r2), now, 0)
                        for fr in fold_rows.itertuples() if pd.notna(fr.r2)]
            client.log_batch(child.info.run_id, metrics=metrics, params=[
                Param("site", r.site), Param("w_xgb", str(r.w_xgb)), Param("selected", str(bool(r.selected)))])
            client.set_terminated(child.info.run_id)
        mlflow.log_artifact(os.path.join(OUT_DIR, "weight_scores.csv"))
        mlflow.log_artifact(os.path.join(OUT_DIR, "proposed_overrides.json"))
        mlflow.log_metric("n_child_runs", len(agg))

    current = _current_weights()
    print(f"{len(agg)} (site, weight) runs logged to MLflow")
    for site, cfg in proposed.items():
        print(f"  {site:14s} current w_xgb={current[site]:.2f} -> proposed {cfg['ensemble_weights'][0]:.2f} "
              f"(score {winners.loc[site, 'score']:.3f})")


# ── 3. One look at the holdout ───────────────────────────────────────────────

def holdout() -> None:
    spark = get_spark("datect-retune-holdout")
    preds = spark.read.format("delta").load(lake_path("gold", "retune_predictions_holdout")).toPandas()
    spark.stop()
    preds = preds[preds["ok"]].copy()
    with open(os.path.join(OUT_DIR, "proposed_overrides.json")) as f:
        proposed = {site: c["ensemble_weights"][0] for site, c in json.load(f).items()}

    result = {"pooled": {}, "per_site": {}}
    for name, weights in (("current", _current_weights()), ("proposed", proposed)):
        p = _ensemble(preds, weights)
        result["pooled"][name] = {"r2": _r2(preds["actual_da"].to_numpy(), p.to_numpy()),
                                  "mae": float(np.abs(preds["actual_da"] - p).mean()), "n": len(preds)}
        for site, g in preds.groupby("site"):
            result["per_site"].setdefault(site, {})[name] = _r2(g["actual_da"].to_numpy(), p.loc[g.index].to_numpy())
    with open(os.path.join(OUT_DIR, "holdout_check.json"), "w") as f:
        json.dump(_nan_to_none(result), f, indent=2, allow_nan=False)

    mlflow = get_mlflow(EXPERIMENT)
    with mlflow.start_run(run_name="holdout-check"):
        mlflow.log_param("git_sha", git_sha(REPO_ROOT))
        mlflow.log_metrics({f"holdout_r2_{k}": v["r2"] for k, v in result["pooled"].items()})
        mlflow.log_artifact(os.path.join(OUT_DIR, "holdout_check.json"))
    print(json.dumps(result["pooled"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--window", choices=["tuning", "validation", "holdout"], required=True)
    p.add_argument("--limit", type=int, help="only the first N points (smoke test)")
    sub.add_parser("score")
    sub.add_parser("holdout")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    if args.cmd == "predict":
        predict(args.window, args.limit)
    elif args.cmd == "score":
        score()
    else:
        holdout()


if __name__ == "__main__":
    main()

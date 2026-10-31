"""
Re-tune per-site ensemble weights and clipping on Spark, every run tracked in MLflow.

Weights and clipping are post-processing: a test point's XGBoost and Random
Forest models are the same whatever those settings are. So instead of re-running
the whole retrospective for every configuration (scripts/eval/
grid_search_weights_clip.py runs 2,250 full retrospectives), this:

1. ``predict --window tuning|holdout``: one Spark task per real DA
   measurement in the window. Each task trains that point's models exactly as
   the retrospective evaluation does (RawForecastEngine._run_single_retrospective)
   with clipping switched off, and returns the raw XGB and RF predictions plus
   the 0.95/0.97/0.99 quantiles and the maximum of its training targets. Rows
   go to Delta.
2. ``score``: applies the 100 weight x clip configurations to the tuning rows
   in Spark SQL and scores each (site, configuration) on yearly folds
   2015-2021. The original grid search used only 2019-2021, but that leaves
   11-32 points per site and fold (none at Cannon Beach in 2019-2020); seven
   folds give 1,581 points instead of 635 and stay before the holdout. Each
   (site, configuration) is an MLflow run; the winner maximizes
   mean R2 - 0.5 * std R2 over folds.
3. ``holdout``: scores the chosen and the current configurations once on the
   2022-2023 holdout rows. Nothing is chosen there.

``predict`` and ``holdout`` also check parity: the current configuration rebuilt
from the raw predictions must equal the committed chronological predictions.

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
import subprocess
import time

import numpy as np
import pandas as pd

from datect_spark.session import get_spark, lake_path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(REPO_ROOT, "results", "tuning_provenance", "retune_spark")

W_XGB_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
# 1.0 clips at the largest training target, which the engine supports as-is
# (np.quantile(y, 1.0)): in practice no quantile clip
CLIP_Q_GRID = [0.95, 0.97, 0.99, 1.0]
CLIP_MAX_GRID = [None, 60.0, 80.0, 100.0, 120.0]

PRED_SCHEMA = (
    "site string, date string, actual_da double, xgb_raw double, rf_raw double, "
    "q95 double, q97 double, q99 double, q100 double, n_train int, task_seconds double, "
    "host string, ok boolean"
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


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True).stdout.strip()


def _mlflow():
    import mlflow

    mlflow.set_tracking_uri(os.environ.get(
        "MLFLOW_TRACKING_URI", "sqlite:///" + os.path.join(os.path.dirname(lake_path()), "mlflow.db")))
    name = "datect-retune"
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=os.path.join(os.path.dirname(lake_path()), "mlartifacts"))
    mlflow.set_experiment(name)
    return mlflow


# ── 1. Distributed training ──────────────────────────────────────────────────

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
                y = out["_internals"]["train_data"]["da_raw"].astype(float).to_numpy()
                row.update(xgb_raw=out["predicted_da"], rf_raw=out["predicted_da_rf"],
                           q95=float(np.quantile(y, 0.95)), q97=float(np.quantile(y, 0.97)),
                           q99=float(np.quantile(y, 0.99)), q100=float(np.quantile(y, 1.0)),
                           n_train=len(y))
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
    parity = _parity(preds, window)

    mlflow = _mlflow()
    with mlflow.start_run(run_name=f"predict-{window}") as parent:
        mlflow.log_params({"window": window, "start": start, "end": end, "git_sha": _git_sha(),
                           "spark_master": sc.master, "default_parallelism": sc.defaultParallelism,
                           "delta_table": table})
        mlflow.log_metrics({"n_points": len(preds), "n_ok": int(preds["ok"].sum()),
                            "wall_seconds": wall, "task_seconds_total": float(preds["task_seconds"].sum()),
                            "hosts": preds["host"].nunique(), **parity})
        n_runs = _log_forecast_runs(parent, preds)
        mlflow.log_metric("n_child_runs", n_runs)
    print(f"{window}: {int(preds['ok'].sum())}/{len(preds)} points in {wall:.0f}s wall "
          f"({preds['task_seconds'].sum():.0f}s of task time on {preds['host'].nunique()} host(s)); "
          f"parity {parity}; {n_runs} forecast runs logged to MLflow")
    spark.stop()


def _log_forecast_runs(parent, preds: pd.DataFrame) -> int:
    """One MLflow child run per trained forecast (XGBoost + Random Forest pair)."""
    from mlflow.entities import Metric, Param
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    ok = preds[preds["ok"]].copy()
    ok["ensemble"] = _ensemble(ok, _current_configs())
    now = int(time.time() * 1000)
    for r in ok.itertuples(index=False):
        child = client.create_run(parent.info.experiment_id, run_name=f"{r.site}|{r.date}",
                                  tags={"mlflow.parentRunId": parent.info.run_id,
                                        "site": r.site, "host": r.host})
        values = {"actual_da": r.actual_da, "xgb_raw": r.xgb_raw, "rf_raw": r.rf_raw,
                  "ensemble_current": r.ensemble, "abs_error_current": abs(r.actual_da - r.ensemble),
                  "train_q99": r.q99, "task_seconds": r.task_seconds}
        client.log_batch(child.info.run_id,
                         metrics=[Metric(k, float(v), now, 0) for k, v in values.items()],
                         params=[Param("site", r.site), Param("test_date", r.date),
                                 Param("n_train", str(r.n_train))])
        client.set_terminated(child.info.run_id)
    return len(ok)


# ── Applying a configuration to raw predictions (pandas, for parity/holdout) ─

def _apply(preds: pd.DataFrame, cfg: dict) -> np.ndarray:
    """The engine's post-processing: per-model clip at the training quantile and
    optional hard ceiling, then the weighted blend."""
    q = {0.95: preds["q95"], 0.97: preds["q97"], 0.99: preds["q99"],
         1.0: preds["q100"]}[cfg["prediction_clip_q"]]
    cap = cfg.get("prediction_clip_max")

    def clip(x):
        x = np.minimum(x, q)
        return np.minimum(x, cap) if cap is not None else x

    w_xgb, w_rf = cfg["ensemble_weights"][:2]
    return (w_xgb * clip(preds["xgb_raw"]) + w_rf * clip(preds["rf_raw"])).to_numpy()


def _current_configs() -> dict:
    from forecasting.tuned_config import get_global, get_per_site

    default_q = get_global()["prediction_clip_q_default"]
    return {site: {"ensemble_weights": c["ensemble_weights"],
                   "prediction_clip_q": c.get("prediction_clip_q") or default_q,
                   "prediction_clip_max": c.get("prediction_clip_max")}
            for site, c in get_per_site().items()}


def _ensemble(preds: pd.DataFrame, configs: dict) -> pd.Series:
    out = pd.Series(np.nan, index=preds.index)
    for site, idx in preds.groupby("site").groups.items():
        out.loc[idx] = _apply(preds.loc[idx], configs[site])
    return out


def _parity(preds: pd.DataFrame, window: str) -> dict:
    """Current configuration rebuilt from raw predictions vs the committed run."""
    if window == "tuning":  # parity is checked on the part with a committed run
        start = pd.Timestamp(_window_bounds("validation")[0])
        return _parity(preds[pd.to_datetime(preds["date"]) >= start], "validation")
    ref = pd.read_parquet(os.path.join(REPO_ROOT, CHRONO[window]))
    ref["date"] = pd.to_datetime(ref["date"])
    ok = preds[preds["ok"]].copy()
    ok["date"] = pd.to_datetime(ok["date"])
    ok["rebuilt"] = _ensemble(ok, _current_configs())
    m = ok.merge(ref[["site", "date", "predicted_da"]], on=["site", "date"], how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    return {"parity_rows_matched": len(both), "parity_rows_unmatched": int((m["_merge"] != "both").sum()),
            "parity_max_abs_diff": float(np.abs(both["rebuilt"] - both["predicted_da"]).max())}


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    sst = float(((y - y.mean()) ** 2).sum())
    return float("nan") if sst == 0 else 1.0 - float(((y - p) ** 2).sum()) / sst


# ── 2. Grid scoring in Spark SQL, one MLflow run per (site, configuration) ──

def score() -> None:
    from pyspark.sql import functions as F

    spark = get_spark("datect-retune-score")
    preds = spark.read.format("delta").load(lake_path("gold", "retune_predictions_tuning")).where("ok")
    grid = spark.createDataFrame(
        [(w, q, m) for w in W_XGB_GRID for q in CLIP_Q_GRID for m in CLIP_MAX_GRID],
        "w_xgb double, clip_q double, clip_max double")
    q = (F.when(F.col("clip_q") == 0.95, F.col("q95"))
         .when(F.col("clip_q") == 0.97, F.col("q97"))
         .when(F.col("clip_q") == 0.99, F.col("q99")).otherwise(F.col("q100")))

    def clip(c):
        # least() skips nulls, so clip_max = null means no hard ceiling
        return F.least(c, q, F.col("clip_max"))

    scored = (preds.crossJoin(grid)
              .withColumn("pred", F.col("w_xgb") * clip(F.col("xgb_raw"))
                          + (1 - F.col("w_xgb")) * clip(F.col("rf_raw")))
              .withColumn("fold", F.year("date"))
              .groupBy("site", "w_xgb", "clip_q", "clip_max", "fold")
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

    # mean - 0.5 * std over the folds that could be scored (as the original grid search)
    folds["clip_max_key"] = folds["clip_max"].fillna(-1.0)
    agg = (folds.dropna(subset=["r2"]).groupby(["site", "w_xgb", "clip_q", "clip_max_key"])
           .agg(mean_r2=("r2", "mean"), std_r2=("r2", lambda s: float(np.std(s))),
                n_folds=("r2", "size"), n=("n", "sum")).reset_index())
    agg["score"] = agg["mean_r2"] - 0.5 * agg["std_r2"]
    agg["clip_max"] = agg["clip_max_key"].where(agg["clip_max_key"] >= 0)
    # Ties (e.g. a ceiling above every prediction) go to the simpler setting:
    # no hard ceiling, then the loosest quantile, then more XGBoost weight.
    agg["has_max"] = agg["clip_max"].notna()
    agg["score_key"] = agg["score"].round(10)  # Spark sums in any order; equal configs differ at ~1e-16
    ranked = agg.sort_values(["site", "score_key", "has_max", "clip_q", "w_xgb"],
                             ascending=[True, False, True, False, False])
    best = ranked.groupby("site").head(1)
    agg["selected"] = agg.index.isin(best.index)
    winners = best.set_index("site")

    current = _current_configs()
    proposed = {site: {"ensemble_weights": [float(r.w_xgb), float(1 - r.w_xgb), 0.0],
                       "prediction_clip_q": float(r.clip_q),
                       "prediction_clip_max": None if pd.isna(r.clip_max) else float(r.clip_max)}
                for site, r in winners.iterrows()}

    os.makedirs(OUT_DIR, exist_ok=True)
    agg.drop(columns=["clip_max_key", "has_max", "score_key"]).to_csv(os.path.join(OUT_DIR, "config_scores.csv"), index=False)
    with open(os.path.join(OUT_DIR, "proposed_overrides.json"), "w") as f:
        json.dump(proposed, f, indent=2)

    mlflow = _mlflow()
    from mlflow.entities import Metric, Param
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    with mlflow.start_run(run_name="score-tuning-folds") as parent:
        mlflow.log_params({"git_sha": _git_sha(), "objective": "mean_r2 - 0.5*std_r2",
                           "folds": "2015-2021 yearly", "n_configs": len(W_XGB_GRID) * len(CLIP_Q_GRID) * len(CLIP_MAX_GRID)})
        now = int(time.time() * 1000)
        exp_id = parent.info.experiment_id
        for _, r in agg.iterrows():
            child = client.create_run(exp_id, run_name=f"{r.site}|w{r.w_xgb}|q{r.clip_q}|max{r.clip_max}",
                                      tags={"mlflow.parentRunId": parent.info.run_id, "site": r.site})
            fold_rows = folds[(folds["site"] == r.site) & (folds["w_xgb"] == r.w_xgb)
                              & (folds["clip_q"] == r.clip_q) & (folds["clip_max_key"] == r.clip_max_key)]
            metrics = [Metric(k, float(v), now, 0) for k, v in
                       (("mean_r2", r.mean_r2), ("std_r2", r.std_r2), ("score", r.score), ("n", r.n))]
            metrics += [Metric(f"r2_{int(fr.fold)}", float(fr.r2), now, 0)
                        for fr in fold_rows.itertuples() if pd.notna(fr.r2)]
            client.log_batch(child.info.run_id, metrics=metrics, params=[
                Param("site", r.site), Param("w_xgb", str(r.w_xgb)), Param("clip_q", str(r.clip_q)),
                Param("clip_max", "none" if pd.isna(r.clip_max) else str(r.clip_max)),
                Param("selected", str(bool(r.selected)))])
            client.set_terminated(child.info.run_id)
        mlflow.log_artifact(os.path.join(OUT_DIR, "config_scores.csv"))
        mlflow.log_artifact(os.path.join(OUT_DIR, "proposed_overrides.json"))
        mlflow.log_metric("n_child_runs", len(agg))

    print(f"{len(agg)} (site, configuration) runs logged to MLflow")
    for site, cfg in proposed.items():
        cur = current[site]
        print(f"  {site:14s} current w={cur['ensemble_weights'][0]:.2f} q={cur['prediction_clip_q']} "
              f"max={cur['prediction_clip_max']} -> w={cfg['ensemble_weights'][0]:.2f} "
              f"q={cfg['prediction_clip_q']} max={cfg['prediction_clip_max']} "
              f"(score {winners.loc[site, 'score']:.3f})")


# ── 3. One look at the holdout ───────────────────────────────────────────────

def holdout() -> None:
    spark = get_spark("datect-retune-holdout")
    preds = spark.read.format("delta").load(lake_path("gold", "retune_predictions_holdout")).toPandas()
    spark.stop()
    preds = preds[preds["ok"]].copy()
    with open(os.path.join(OUT_DIR, "proposed_overrides.json")) as f:
        proposed = json.load(f)
    current = _current_configs()

    result = {"pooled": {}, "per_site": {}}
    for name, cfgs in (("current", current), ("proposed", proposed)):
        p = _ensemble(preds, cfgs)
        result["pooled"][name] = {"r2": _r2(preds["actual_da"].to_numpy(), p.to_numpy()),
                                  "mae": float(np.abs(preds["actual_da"] - p).mean()), "n": len(preds)}
        for site, g in preds.groupby("site"):
            result["per_site"].setdefault(site, {})[name] = _r2(g["actual_da"].to_numpy(), p.loc[g.index].to_numpy())
    with open(os.path.join(OUT_DIR, "holdout_check.json"), "w") as f:
        json.dump(result, f, indent=2)

    mlflow = _mlflow()
    with mlflow.start_run(run_name="holdout-check"):
        mlflow.log_param("git_sha", _git_sha())
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

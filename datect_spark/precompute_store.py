"""
Precompute the dashboard's retrospective, spectral and correlation results on Spark.

These endpoints used to be served from a file cache (precompute_cache.py, removed).
They now come from three Delta tables of the precomputed store (backend/precomputed.py),
next to gold/forecasts (datect_spark/forecast_serving.py):

    retrospective   run_retrospective_evaluation for the two slices the spectral
                    analysis reads (regression/xgboost and regression/rf). Spark
                    runs the engine's own per-point forecasts: the engine fans them out
                    through joblib, and here joblib's backend is Spark (joblibspark,
                    DATECT_PARALLEL_BACKEND=spark), so sampling and assembly are the
                    engine's, unchanged. Rows hold the retrospective records (backend/records.py).
    spectral        generate_spectral_analysis per site and for all sites, with the
                    regression/xgboost and regression/rf slices as model traces
    correlation     generate_correlation_heatmap per site

Each row carries the config hash; the API serves only rows matching its own.

Usage (Hyak, inside an allocation; see datect_spark/store.sbatch):
    python -m datect_spark.precompute_store retrospective
    python -m datect_spark.precompute_store visuals
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time

import pandas as pd

from backend.precomputed import SPECTRAL_SLICES, SPECTRAL_TRACE
from datect_spark.session import get_mlflow, get_spark, git_sha, lake_path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(REPO_ROOT, "results", "store")
EXPERIMENT = "datect-serving"
# Only the spectral analysis reads retrospective records (the XGBoost and RF model traces)
SLICES = SPECTRAL_SLICES


def _spark(name: str):
    os.environ["PYTHONPATH"] = REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
    return get_spark(name, conf={"spark.driver.memory": "8g"})


def _write(spark, rows: list[dict], table: str) -> str:
    if not rows:
        raise SystemExit(f"nothing to write to {table}")
    path = lake_path("gold", table)
    spark.createDataFrame(pd.DataFrame(rows)).coalesce(1).write.format("delta") \
        .mode("overwrite").option("overwriteSchema", "true").save(path)
    return path


def _stamp(rows: list[dict], chash: str, sha: str) -> list[dict]:
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return [{**r, "config_hash": chash, "git_sha": sha, "computed_at": now} for r in rows]


def retrospective(max_test_samples: int | None = None) -> dict:
    # Must be set before config is imported: the engine's joblib calls go to Spark
    os.environ["DATECT_PARALLEL_BACKEND"] = "spark"
    from joblibspark import register_spark

    from backend.api import get_forecast_engine
    from backend.precomputed import config_hash
    from backend.records import retrospective_records

    spark = _spark("datect-store-retrospective")
    register_spark()
    engine = get_forecast_engine()
    rows, timings = [], {}
    for task, model in SLICES:
        t0 = time.time()
        results_df = engine.run_retrospective_evaluation(task=task, model_type=model,
                                                         max_test_samples=max_test_samples)
        seconds = time.time() - t0
        records = retrospective_records(results_df) if results_df is not None and not results_df.empty else None
        timings[f"{task}/{model}"] = {"seconds": seconds, "records": len(records or [])}
        print(f"  {task}/{model}: {len(records or [])} records in {seconds:.0f}s", flush=True)
        if records is None:
            continue  # nothing to serve: on demand the API reports it too
        # No default=str: a type json cannot write must fail here, not be stringified
        rows.append({"task": task, "model_type": model, "n_records": len(records),
                     "records_json": json.dumps(records)})
    chash = config_hash()
    path = _write(spark, _stamp(rows, chash, git_sha(REPO_ROOT)), "retrospective")
    spark.stop()
    return {"table": path, "config_hash": chash, "max_test_samples": max_test_samples,
            "slices": timings, "total_seconds": sum(t["seconds"] for t in timings.values())}


def _fresh_data() -> pd.DataFrame:
    import config

    os.chdir(REPO_ROOT)  # config paths are relative to the repo root
    return pd.read_parquet(config.FINAL_OUTPUT_PATH)


def _spectral(site, xgb_results, rf_results):
    """A site's spectral plots, or None when it fails (not stored, as before)."""
    from backend.visualizations import generate_spectral_analysis

    try:
        return generate_spectral_analysis(_fresh_data(), site, xgb_results, rf_results)
    except Exception as exc:
        print(f"spectral {site}: {exc}")
        return None


def _correlation(site):
    """A site's correlation heatmap, or None when it fails (not stored, as before)."""
    from backend.visualizations import generate_correlation_heatmap

    try:
        return generate_correlation_heatmap(_fresh_data(), site=site)
    except Exception as exc:
        print(f"correlation {site}: {exc}")
        return None


def visuals() -> dict:
    from backend.precomputed import ALL_SITES, Store, config_hash

    chash, sha = config_hash(), git_sha(REPO_ROOT)
    store = Store(lake_path("gold"), expected_hash=chash)
    traces = {}
    for task, model in SPECTRAL_SLICES:
        records = store.retrospective(task, model)
        if records is None:
            raise SystemExit(f"run the retrospective step first: {task}/{model} is missing")
        traces[SPECTRAL_TRACE[model]] = pd.DataFrame(records)
    xgb, rf = traces["xgb"], traces["rf"]

    spark = _spark("datect-store-visuals")
    sc = spark.sparkContext
    sites = sorted(_fresh_data()["site"].unique())
    t0 = time.time()
    # One Spark task per chart; an empty or failed chart is not stored (computed on demand)
    spectral = (sc.parallelize(sites + [None], len(sites) + 1)
                .map(lambda s: (s or ALL_SITES, _spectral(s, xgb, rf)))
                .filter(lambda kv: bool(kv[1]))
                .map(lambda kv: {"site": kv[0], "plots_json": json.dumps(kv[1])}).collect())
    correlation = (sc.parallelize(sites, len(sites))
                   .map(lambda s: (s, _correlation(s)))
                   .filter(lambda kv: kv[1] is not None)
                   .map(lambda kv: {"site": kv[0], "plot_json": json.dumps(kv[1])}).collect())
    seconds = time.time() - t0
    paths = {"spectral": _write(spark, _stamp(spectral, chash, sha), "spectral"),
             "correlation": _write(spark, _stamp(correlation, chash, sha), "correlation")}
    spark.stop()
    return {"tables": paths, "config_hash": chash, "spectral_rows": len(spectral),
            "correlation_rows": len(correlation), "seconds": seconds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("step", choices=["retrospective", "visuals"])
    parser.add_argument("--max-test-samples", type=int, help="smoke runs: points per slice")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    t0 = time.time()
    out = retrospective(args.max_test_samples) if args.step == "retrospective" else visuals()
    out = {"step": args.step, "git_sha": git_sha(REPO_ROOT), "host": socket.gethostname(),
           "cpus": os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count())),
           "wall_seconds": time.time() - t0, **out}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"precompute_{args.step}.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    try:  # the JSON above is the record; tracking must not stop the build before `save`
        mlflow = get_mlflow(EXPERIMENT)
        with mlflow.start_run(run_name=f"store-{args.step}"):
            mlflow.log_param("git_sha", out["git_sha"])
            mlflow.log_metric("wall_seconds", out["wall_seconds"])
    except Exception as exc:
        print(f"MLflow logging failed: {exc}")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

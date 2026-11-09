"""
Precompute the dashboard's forecasts on Spark into Delta, registered in MLflow.

A dashboard forecast (POST /api/forecast/enhanced) is one regression
generate_single_forecast call, which trains the evaluated models on data up to the
anchor (categories come from the error band, backend/band.py). ``precompute`` runs those calls for every (site, date) on the grid
(one Spark task per pair, one thread each) and stores each output as-is
(lossless JSON, backend/precomputed.encode) in ``gold/forecasts`` with the
registered model version and the config hash. The API loads that table with
deltalake (no JVM) and falls back to computing on demand on a miss.

Grids: ``weekly`` = the (site, Monday) rows of final_output.parquet; ``daily`` = every
day from the first to the last row, per site. Models: ``ensemble`` (the dashboard
default).

``parity`` checks that sampled table entries equal a fresh on-demand computation
exactly (values and types), in a process with the API's threading, and that the
registered pyfunc model reproduces the table.

Usage (Hyak, inside an allocation; see datect_spark/serving.sbatch):
    python -m datect_spark.forecast_serving precompute [--grid weekly|daily] [--limit N]
    python -m datect_spark.forecast_serving parity [--n 50]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from backend.precomputed import FORECAST_TASKS
from datect_spark.session import get_mlflow, get_spark, git_sha, lake_path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(REPO_ROOT, "results", "serving")
TABLE = ("gold", "forecasts")
REGISTERED_MODEL = "datect-forecaster"
EXPERIMENT = "datect-serving"
TASKS = FORECAST_TASKS  # backend/precomputed.py: what the API reads
MODELS = ("ensemble",)
OUT_SCHEMA = (
    "site string, forecast_date date, task string, model_type string, "
    "predicted_da double, predicted_category int, spike_probability double, spike_alert boolean, "
    "training_samples int, result_json string, error string, task_seconds double, host string"
)


def build_grid(grid: str) -> list[tuple[str, str]]:
    """(site, YYYY-MM-DD) pairs, latest first (most training rows: longest tasks)."""
    data = pd.read_parquet(os.path.join(REPO_ROOT, "data", "processed", "final_output.parquet"),
                           columns=["date", "site"])
    data["date"] = pd.to_datetime(data["date"])
    if grid == "weekly":
        pairs = data.drop_duplicates(["site", "date"])
    else:
        pairs = pd.concat(pd.DataFrame({"site": site, "date": pd.date_range(g.min(), g.max(), freq="D")})
                          for site, g in data.groupby("site")["date"])
    pairs = pairs.sort_values(["date", "site"], ascending=[False, True])
    return [(s, d.strftime("%Y-%m-%d")) for s, d in zip(pairs["site"], pairs["date"])]


# ── Spark task ───────────────────────────────────────────────────────────────

_ENGINE = None


def _engine():
    """One engine (feature frame cached) per Python worker, as the API keeps one."""
    global _ENGINE
    if _ENGINE is None:
        os.chdir(REPO_ROOT)  # config paths are relative to the repo root
        import config
        from forecasting import ForecastEngine

        engine = ForecastEngine(validate_on_init=False)
        engine._load_feature_frame()
        _ENGINE = (engine, config)
    return _ENGINE


def _summary(result: dict | None) -> dict:
    if result is None:
        return {}
    return {"predicted_da": result.get("predicted_da"),
            "predicted_category": result.get("predicted_category"),
            "spike_probability": result.get("spike_probability"),
            "spike_alert": result.get("spike_alert"),
            "training_samples": result.get("training_samples")}


def _forecast_partition(batches):
    from backend.precomputed import encode

    engine, config = _engine()
    host = socket.gethostname()
    columns = [c.split()[0] for c in OUT_SCHEMA.split(", ")]
    for pdf in batches:
        rows = []
        for r in pdf.itertuples(index=False):
            for task in TASKS:
                t0 = time.time()
                model = r.model_type
                row = {"site": r.site, "forecast_date": pd.Timestamp(r.date).date(), "task": task,
                       "model_type": model, "host": host}
                try:
                    result = engine.generate_single_forecast(
                        config.FINAL_OUTPUT_PATH, pd.Timestamp(r.date), r.site, task, model)
                    row.update(_summary(result), result_json=encode(result))
                except Exception as exc:  # the API answers these with an error: not served
                    row["error"] = f"{type(exc).__name__}: {exc}"
                row["task_seconds"] = time.time() - t0
                rows.append(row)
        out = pd.DataFrame(rows).reindex(columns=columns)
        for col in ("predicted_category", "training_samples"):
            out[col] = out[col].astype("Int32")
        out["spike_alert"] = out["spike_alert"].astype("boolean")
        for col in ("result_json", "error"):  # None, not NaN, for Arrow strings
            out[col] = out[col].astype(object).where(out[col].notna(), None)
        yield out


# ── MLflow Model Registry ────────────────────────────────────────────────────

def _forecaster_class():
    import mlflow.pyfunc

    class DatectForecaster(mlflow.pyfunc.PythonModel):
        """The procedure behind gold/forecasts: rows (site, date, task, model_type) ->
        generate_single_forecast output as lossless JSON. It retrains per request on
        data up to the anchor (that is the model), so it runs inside a DATECT checkout."""

        def predict(self, context, model_input, params=None):
            from backend.precomputed import encode

            engine, config = _engine()
            return [encode(engine.generate_single_forecast(
                config.FINAL_OUTPUT_PATH, pd.Timestamp(r.date), r.site, r.task, r.model_type))
                for r in model_input.itertuples(index=False)]

    return DatectForecaster


def register_model(mlflow, tags: dict) -> str:
    import scipy
    import sklearn
    import xgboost

    info = mlflow.pyfunc.log_model(
        name="forecaster", python_model=_forecaster_class()(),
        artifacts={"tuned_hyperparameters": os.path.join(REPO_ROOT, "config", "tuned_hyperparameters.json")},
        pip_requirements=[f"{m.__name__ if m is not sklearn else 'scikit-learn'}=={m.__version__}"
                          for m in (np, pd, sklearn, xgboost, scipy)],
        registered_model_name=REGISTERED_MODEL, metadata=tags)
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    version = getattr(info, "registered_model_version", None)
    if version is None:
        version = max(int(v.version) for v in client.search_model_versions(f"name='{REGISTERED_MODEL}'"))
    version = str(version)
    for k, v in tags.items():
        client.set_model_version_tag(REGISTERED_MODEL, version, k, str(v))
    return version


# ── Steps ────────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    return git_sha(REPO_ROOT)


def precompute(grid: str, limit: int | None) -> dict:
    from pyspark.sql import functions as F

    from backend.precomputed import config_hash

    pairs = build_grid(grid)
    if limit:
        pairs = pairs[:limit]  # smoke runs
    chash, sha = config_hash(), _git_sha()
    table = lake_path(*TABLE)
    mlflow = get_mlflow(EXPERIMENT)
    with mlflow.start_run(run_name=f"precompute-{grid}"):
        tags = {"config_hash": chash, "git_sha": sha, "grid": grid, "models": ",".join(MODELS),
                "delta_table": table}
        mlflow.log_params(tags)
        version = register_model(mlflow, tags)
        model_version = f"{REGISTERED_MODEL}/{version}"

        os.environ["PYTHONPATH"] = REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
        spark = get_spark(f"datect-precompute-{grid}", conf={"spark.driver.memory": "8g"})
        sc = spark.sparkContext
        records = [(site, day, model) for site, day in pairs for model in MODELS]
        t0 = time.time()
        tasks = spark.createDataFrame(sc.parallelize(records, numSlices=len(records)),
                                      "site string, date string, model_type string")
        (tasks.mapInPandas(_forecast_partition, schema=OUT_SCHEMA)
         .withColumn("model_version", F.lit(model_version))
         .withColumn("config_hash", F.lit(chash))
         .withColumn("git_sha", F.lit(sha))
         .withColumn("computed_at", F.current_timestamp())
         .repartition(4)  # after the shuffle: the forecasts still run one task per pair
         .write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(table))
        wall = time.time() - t0

        from delta.tables import DeltaTable

        rows = spark.read.format("delta").load(table)
        stats = rows.agg(F.count("*").alias("entries"),
                         F.sum(F.col("error").isNotNull().cast("int")).alias("errors"),
                         F.sum((F.col("error").isNull() & F.col("predicted_da").isNull()).cast("int"))
                         .alias("no_forecast"),
                         F.sum("task_seconds").alias("task_seconds"),
                         F.countDistinct("host").alias("hosts")).first().asDict()
        detail = spark.sql(f"DESCRIBE DETAIL delta.`{table}`").first().asDict()
        delta_version = DeltaTable.forPath(spark, table).history(1).first()["version"]
        spark.stop()

        cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
        per_pair = stats["task_seconds"] / len(records)
        daily_pairs = len(build_grid("daily"))
        out = {"grid": grid, "pairs": len(pairs), "models": list(MODELS), "entries": stats["entries"],
               "errors": stats["errors"], "entries_without_forecast": stats["no_forecast"], "wall_seconds": wall,
               "task_seconds_total": stats["task_seconds"], "task_seconds_per_pair": per_pair,
               "cores": cores, "hosts": stats["hosts"], "task_seconds_over_wall": stats["task_seconds"] / wall,
               "daily_pairs": daily_pairs,
               "daily_estimate_node_hours": per_pair * daily_pairs * len(MODELS) / cores / 3600,
               "table": table, "delta_version": delta_version, "table_files": detail["numFiles"],
               "table_bytes": detail["sizeInBytes"], "model_version": model_version,
               "config_hash": chash, "git_sha": sha}
        mlflow.log_metrics({k: float(v) for k, v in out.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)})
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        for k in ("delta_version", "entries", "wall_seconds"):
            client.set_model_version_tag(REGISTERED_MODEL, version, k, str(out[k]))
    return out


def parity(n: int) -> dict:
    """n random table entries vs generate_single_forecast run fresh, in a process with
    the API's threading (session.py pins one thread per Spark task; the API does not)."""
    from backend.precomputed import Store, decode, same

    store = Store(lake_path("gold"))
    keys = sorted(store._index["forecasts"])
    picks = [keys[i] for i in np.random.default_rng(0).choice(len(keys), size=min(n, len(keys)), replace=False)]
    child = (
        "import json, os, sys, pandas as pd\n"
        "import config\n"
        "from forecasting import ForecastEngine\n"
        "from backend.precomputed import encode\n"
        "engine = ForecastEngine(validate_on_init=False)\n"
        "for site, day, task, model in json.load(sys.stdin):\n"
        "    r = engine.generate_single_forecast(config.FINAL_OUTPUT_PATH, pd.Timestamp(day), site, task, model)\n"
        "    print('PARITY ' + json.dumps(encode(r)), flush=True)\n")
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    env["PYTHONPATH"] = REPO_ROOT
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-c", child], input=json.dumps(picks), cwd=REPO_ROOT,
                          env=env, capture_output=True, text=True, check=True)
    on_demand = [decode(json.loads(line[len("PARITY "):]))
                 for line in proc.stdout.splitlines() if line.startswith("PARITY ")]
    assert len(on_demand) == len(picks)
    seconds = time.time() - t0
    checks = []
    for key, fresh in zip(picks, on_demand):
        hit = store.forecast(*key)
        checks.append({"key": list(key), "equal": same(hit.result, fresh), "has_forecast": fresh is not None})
    table = store.tables["forecasts"]
    out = {"n": len(checks), "equal": sum(c["equal"] for c in checks),
           "with_forecast": sum(c["has_forecast"] for c in checks),
           "on_demand_seconds": seconds, "table_version": table["version"],
           "table_entries": table["entries"], "stale_rows": table["stale_rows"], "checks": checks}

    # The registered model reproduces the table too
    mlflow = get_mlflow(EXPERIMENT)
    from mlflow import pyfunc

    version = store.forecast(*picks[0]).model_version.split("/")[1]
    model = pyfunc.load_model(f"models:/{REGISTERED_MODEL}/{version}")
    few = picks[:3]
    predicted = model.predict(pd.DataFrame(few, columns=["site", "date", "task", "model_type"]))
    out["registry_model"] = {"version": version, "n": len(few),
                             "equal": sum(same(decode(p), store.forecast(*k).result)
                                          for p, k in zip(predicted, few))}
    out["passed"] = out["equal"] == out["n"] and out["registry_model"]["equal"] == len(few)
    with mlflow.start_run(run_name="parity"):
        mlflow.log_metrics({"parity_n": out["n"], "parity_equal": out["equal"],
                            "registry_equal": out["registry_model"]["equal"]})
    if not out["passed"]:
        print(json.dumps(out, indent=2, default=str))
        raise SystemExit("parity failed")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("precompute")
    p.add_argument("--grid", choices=["weekly", "daily"], default="weekly")
    p.add_argument("--limit", type=int, help="only the first N (site, date) pairs (smoke test)")
    q = sub.add_parser("parity")
    q.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    t0 = time.time()
    out = precompute(args.grid, args.limit) if args.cmd == "precompute" else parity(args.n)
    out = {"step": args.cmd, "host": socket.gethostname(), "step_seconds": time.time() - t0, **out}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{args.cmd}.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "checks"}, indent=2, default=str))


if __name__ == "__main__":
    main()

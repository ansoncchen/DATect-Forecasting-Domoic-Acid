"""SparkSession with Delta Lake, configured for Hyak.

Paths come from environment variables so the same code runs on a laptop and on
Hyak, where the lake and Spark's scratch belong on /gscratch/scrubbed (the stf
group allocation is nearly full):

    DATECT_LAKE          Delta tables root     (default ./lake)
    SPARK_LOCAL_DIRS     shuffle/spill scratch (Spark reads this itself)
    DATECT_IVY_DIR       Maven/Ivy jar cache   (default <lake>/../ivy)
    SPARK_MASTER         e.g. spark://host:7077; default local[*]
"""
from __future__ import annotations

import os
import time

# One thread per task: Spark runs one Python worker per core, and models already
# train with n_jobs=1. Must be set before any worker starts.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
# PySpark converts naive datetimes with the process's local time zone; Hyak runs
# on Pacific time, which would shift every date. Pin everything to UTC.
os.environ["TZ"] = "UTC"
time.tzset()


def lake_path(*parts: str) -> str:
    root = os.environ.get("DATECT_LAKE", os.path.abspath("lake"))
    return os.path.join(root, *parts)


def get_spark(app_name: str, master: str | None = None, conf: dict | None = None):
    """A SparkSession with the Delta extensions; ``conf`` adds or overrides settings."""
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    lake = os.environ.get("DATECT_LAKE", os.path.abspath("lake"))
    ivy = os.environ.get("DATECT_IVY_DIR", os.path.join(os.path.dirname(lake), "ivy"))
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master or os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.jars.ivy", ivy)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.python.worker.reuse", "true")
    )
    for key, value in (conf or {}).items():
        builder = builder.config(key, value)
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

"""
MODIS imagery into Delta Lake (bronze -> silver -> gold), checked against the cube.

Reads the MODIS 8-day composites already on Hyak (chla, k490, nflh, sst; they are
never re-downloaded) in place and builds:

  bronze/satellite_files    one row per NetCDF: channel, file name, rank in the
                            reference's sorted() order, bytes, sha256, n_times, grid
  bronze/satellite_times    every timestamp in every file, fully cloudy ones too
  bronze/satellite_grid     the lat/lon coordinates (identical in every file)
  bronze/satellite_pixels   every non-NaN pixel: time, file_rank, time_pos, lat_idx,
                            lon_idx, value; partitioned by channel and year
  silver/satellite_times    one source per (channel, time): the first file in sorted
                            order, as the reference's drop_duplicates(keep="first")
  silver/cube_times         times present in all four channels (the cube's time axis)
  silver/satellite_cube     one row per (time, pixel) with any channel present: the
                            four raw values and valid = all four present (by year)
  gold/oad_region_valid_frac  per (date, region): valid pixels / pixels in the region's
                            lat/lon box, land included (regions from oad/src/regions.py)

The reference is oad/scripts/02_build_cube.py followed by
07_cloud_confound.compute_region_valid_fractions, which produced
data/processed/oad_cloud_fractions.parquet. ``parity`` checks that the gold table
equals that file exactly, that the time axis equals cube.zarr's, and that five dates
of standardized values and the mask equal cube.zarr. ``reference`` re-runs and times
the reference itself (cube written to scrubbed; the original is never touched).

Each step writes its measurements to results/spark_lake/modis_<step>.json and skips
itself when that file already holds a run of the same commit (ckpt requeues).

Usage (Hyak, inside an allocation; see datect_spark/modis_lake.sbatch):
    python -m datect_spark.modis_lake bronze|silver|optimize|gold|parity|vacuum|reference
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from datect_spark.session import get_mlflow, get_spark, lake_path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(REPO_ROOT, "results", "spark_lake")
RAW_ROOT = os.environ.get("OAD_RAW", "/gscratch/stf/ac283/oad_current/data/raw")
CUBE_PATH = os.environ.get("OAD_CUBE", "/gscratch/stf/ac283/oad_current/data/cube.zarr")
REFERENCE_PARQUET = os.path.join(REPO_ROOT, "data", "processed", "oad_cloud_fractions.parquet")
# Smoke runs: only files of these years (by file name, as 02_build_cube.py --year)
YEARS = {y for y in os.environ.get("DATECT_MODIS_YEARS", "").split(",") if y}

TABLES = {
    "files": ("bronze", "satellite_files"),
    "times": ("bronze", "satellite_times"),
    "grid": ("bronze", "satellite_grid"),
    "pixels": ("bronze", "satellite_pixels"),
    "winners": ("silver", "satellite_times"),
    "cube_times": ("silver", "cube_times"),
    "cube": ("silver", "satellite_cube"),
    "valid_frac": ("gold", "oad_region_valid_frac"),
}

FILE_SCHEMA = "channel string, var string, file_name string, path string, file_rank int, bytes long"
FILE_COLUMNS = ["channel", "var", "file_name", "path", "file_rank", "bytes"]
TIMES_SCHEMA = (
    "channel string, file_name string, file_rank int, bytes long, sha256 string, "
    "n_lat int, n_lon int, grid_sha string, time_pos int, time timestamp, year int"
)
PIXEL_SCHEMA = (
    "channel string, year int, time timestamp, file_rank int, time_pos short, "
    "lat_idx short, lon_idx short, value float"
)
# OPTIMIZE target file size: small enough that one year of the cube spans several
# files, so Z-ordered files cover separate parts of the grid and can be skipped
OPTIMIZE_MAX_FILE_BYTES = int(os.environ.get("DATECT_OPTIMIZE_MAX_FILE_BYTES", 64 * 1024 * 1024))
SPARK_CONF = {
    "spark.driver.memory": os.environ.get("DATECT_SPARK_DRIVER_MEMORY", "16g"),
    "spark.sql.shuffle.partitions": "256",
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    "spark.databricks.delta.optimize.maxFileSize": str(OPTIMIZE_MAX_FILE_BYTES),
    # VACUUM of the files OPTIMIZE replaced, in the same run (scratch lake)
    "spark.databricks.delta.retentionDurationCheck.enabled": "false",
}


def table(key: str) -> str:
    return lake_path(*TABLES[key])


# ── The reference's own definitions (oad/), loaded without shadowing DATECT's config ──

def _load_oad_module(name: str, rel_path: str):
    oad_dir = os.path.join(REPO_ROOT, "oad")
    spec = importlib.util.spec_from_file_location(name, os.path.join(oad_dir, rel_path))
    module = importlib.util.module_from_spec(spec)
    saved_path, saved_config = list(sys.path), sys.modules.pop("config", None)
    try:
        spec.loader.exec_module(module)  # regions.py / 07_*.py put oad/ on sys.path
    finally:
        sys.path[:] = saved_path
        if saved_config is not None:
            sys.modules["config"] = saved_config
        else:
            sys.modules.pop("config", None)
    return module


def oad_channels() -> list[tuple[str, str, bool]]:
    """(name, NetCDF variable, log_transform) in cube channel order."""
    cfg = _load_oad_module("oad_config", "config.py")
    return [(name, var, log) for name, _, var, log in cfg.CHANNELS]


def oad_regions():
    return _load_oad_module("oad_regions", os.path.join("src", "regions.py")).REGIONS


def list_files(raw_root: str = RAW_ROOT) -> tuple[list[tuple], list[str]]:
    """The NetCDFs the reference reads, ranked in its order (sorted glob per channel),
    and the AppleDouble ``._*`` stubs it skips (macOS metadata, not NetCDF)."""
    rows, stubs = [], []
    for channel, var, _ in oad_channels():
        ch_dir = Path(raw_root) / channel
        stubs += sorted(str(p) for p in ch_dir.iterdir() if p.name.startswith("._"))
        for rank, p in enumerate(sorted(ch_dir.glob(f"{channel}_*.nc"))):
            if not YEARS or p.stem.split("_")[1][:4] in YEARS:
                rows.append((channel, var, p.name, str(p), rank, p.stat().st_size))
    return rows, stubs


def channel_array(ds, var: str):
    """One channel as (time, lat, lon), named like the reference's _open_channel_dedup."""
    da = ds[var]
    for d in ("altitude", "depth"):
        if d in da.dims:
            da = da.squeeze(d, drop=True)
    return da.rename({k: v for k, v in (("latitude", "lat"), ("longitude", "lon")) if k in da.dims})


def grid_sha(lat: np.ndarray, lon: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(lat, dtype=np.float64).tobytes()
                          + np.ascontiguousarray(lon, dtype=np.float64).tobytes()).hexdigest()


def region_boxes(lat: np.ndarray, lon: np.ndarray) -> pd.DataFrame:
    """Index ranges of each region's box on the grid, with the reference's pixel count
    (lat_in.sum() * lon_in.sum(), same comparisons as compute_region_valid_fractions)."""
    rows = []
    for order, r in enumerate(oad_regions()):
        lat_in = np.flatnonzero((lat >= r.lat_min) & (lat <= r.lat_max))
        lon_in = np.flatnonzero((lon >= r.lon_min) & (lon <= r.lon_max))
        if len(lat_in) == 0 or len(lon_in) == 0:
            continue  # the reference skips empty regions too
        # monotonic coordinates make each box a contiguous index range
        assert np.array_equal(lat_in, np.arange(lat_in[0], lat_in[-1] + 1))
        assert np.array_equal(lon_in, np.arange(lon_in[0], lon_in[-1] + 1))
        rows.append({"region_order": order, "region": r.name,
                     "lat_lo": int(lat_in[0]), "lat_hi": int(lat_in[-1]),
                     "lon_lo": int(lon_in[0]), "lon_hi": int(lon_in[-1]),
                     "region_pixels": int(len(lat_in) * len(lon_in))})
    return pd.DataFrame(rows)


def standardize(values: np.ndarray, log_transform: bool, mean: float, std: float) -> np.ndarray:
    """02_build_cube's per-channel transform on raw float32 values (NaN stays NaN)."""
    x = values.astype(np.float32)
    if log_transform:
        x = np.log(np.clip(x, 1e-4, None))
    return (x - mean) / (std + 1e-8)


# ── Spark tasks (run in Python workers; one NetCDF per task) ─────────────────

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_times(batches):
    import xarray as xr

    for pdf in batches:
        for r in pdf.itertuples(index=False):
            with xr.open_dataset(r.path, engine="netcdf4") as ds:
                da = channel_array(ds, r.var)
                times = pd.to_datetime(da["time"].values)
                lat, lon = da["lat"].values, da["lon"].values
            yield pd.DataFrame({
                "channel": r.channel, "file_name": r.file_name, "file_rank": r.file_rank,
                "bytes": r.bytes, "sha256": _sha256(r.path), "n_lat": len(lat), "n_lon": len(lon),
                "grid_sha": grid_sha(lat, lon), "time_pos": np.arange(len(times), dtype=np.int32),
                "time": times, "year": times.year.astype(np.int32)})


def _file_pixels(batches):
    import xarray as xr

    for pdf in batches:
        for r in pdf.itertuples(index=False):
            with xr.open_dataset(r.path, engine="netcdf4") as ds:
                da = channel_array(ds, r.var)
                values = da.values  # (time, lat, lon) float32, fill value decoded to NaN
                times = pd.to_datetime(da["time"].values)
            n_lon = values.shape[2]
            for pos in range(values.shape[0]):
                frame = values[pos].ravel()
                flat = np.flatnonzero(~np.isnan(frame))
                yield pd.DataFrame({
                    "channel": r.channel, "year": np.int32(times[pos].year), "time": times[pos],
                    "file_rank": np.int32(r.file_rank), "time_pos": np.int16(pos),
                    "lat_idx": (flat // n_lon).astype(np.int16),
                    "lon_idx": (flat % n_lon).astype(np.int16),
                    "value": frame[flat]})


# ── Delta bookkeeping ────────────────────────────────────────────────────────

def live_files(path: str) -> pd.DataFrame:
    """Files in the latest version of a Delta table, with their row counts and
    min/max statistics, replayed from the JSON commit log."""
    log_dir = Path(path) / "_delta_log"
    live = {}
    for commit in sorted(log_dir.glob("*.json")):
        for line in commit.read_text().splitlines():
            action = json.loads(line)
            if "add" in action:
                a = action["add"]
                stats = json.loads(a.get("stats") or "{}")
                live[a["path"]] = {"path": a["path"], "size": a["size"],
                                   "num_records": stats.get("numRecords"),
                                   **{f"min_{k}": v for k, v in stats.get("minValues", {}).items()},
                                   **{f"max_{k}": v for k, v in stats.get("maxValues", {}).items()}}
            elif "remove" in action:
                live.pop(action["remove"]["path"], None)
    return pd.DataFrame(list(live.values()))


def du_bytes(path: str) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def table_stats(path: str) -> dict:
    files = live_files(path)
    return {"files": len(files), "bytes": int(files["size"].sum()),
            "rows": int(files["num_records"].sum()), "bytes_on_disk": du_bytes(path)}


def files_overlapping(files: pd.DataFrame, box) -> tuple[int, int]:
    """(files, bytes) whose lat_idx/lon_idx min-max range meets the box: what Delta's
    data skipping must still read for a query filtered to that box."""
    hit = ((files["max_lat_idx"] >= box.lat_lo) & (files["min_lat_idx"] <= box.lat_hi)
           & (files["max_lon_idx"] >= box.lon_lo) & (files["min_lon_idx"] <= box.lon_hi))
    return int(hit.sum()), int(files.loc[hit, "size"].sum())


# ── Steps ────────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True).stdout.strip()


def _spark(name: str):
    os.environ["PYTHONPATH"] = REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
    return get_spark(name, conf=SPARK_CONF)


def bronze() -> dict:
    from pyspark.sql import functions as F

    files, stubs = list_files()
    by_size = sorted(files, key=lambda r: -r[5])  # longest tasks first
    spark = _spark("datect-modis-bronze")
    sc = spark.sparkContext
    out = {"raw_root": RAW_ROOT, "years": sorted(YEARS) or "all", "n_files": len(files), "bytes_ingested": int(sum(r[5] for r in files)),
           "bytes_per_channel": pd.DataFrame(files, columns=FILE_COLUMNS)
           .groupby("channel")["bytes"].sum().astype(int).to_dict(),
           "skipped_appledouble_stubs": len(stubs),
           "skipped_appledouble_bytes": int(sum(os.path.getsize(p) for p in stubs))}

    def file_df():
        return spark.createDataFrame(sc.parallelize(by_size, numSlices=len(by_size)), FILE_SCHEMA)

    t0 = time.time()
    times = file_df().mapInPandas(_file_times, schema=TIMES_SCHEMA).cache()
    times.select("channel", "file_name", "file_rank", "time_pos", "time", "year") \
        .write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(table("times"))
    per_file = (times.groupBy("channel", "file_name", "file_rank", "bytes", "sha256", "n_lat", "n_lon", "grid_sha")
                .agg(F.count("*").alias("n_times"), F.min("time").alias("first_time"),
                     F.max("time").alias("last_time")))
    per_file.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(table("files"))
    meta = per_file.toPandas()
    out["metadata_seconds"] = time.time() - t0
    out["n_file_rows"] = len(meta)
    out["n_time_rows"] = int(meta["n_times"].sum())
    grids = meta["grid_sha"].unique()
    out["grids"] = len(grids)
    if len(meta) != len(files) or len(grids) != 1:
        raise RuntimeError(f"{len(meta)} of {len(files)} files read; {len(grids)} distinct grids")

    import xarray as xr

    with xr.open_dataset(by_size[0][3], engine="netcdf4") as ds:
        da = channel_array(ds, by_size[0][1])
        lat, lon = da["lat"].values, da["lon"].values
    assert grid_sha(lat, lon) == grids[0]
    grid = pd.concat([pd.DataFrame({"axis": "lat", "idx": np.arange(len(lat), dtype=np.int32), "coord": lat}),
                      pd.DataFrame({"axis": "lon", "idx": np.arange(len(lon), dtype=np.int32), "coord": lon})])
    spark.createDataFrame(grid).write.format("delta").mode("overwrite").save(table("grid"))

    t0 = time.time()
    file_df().mapInPandas(_file_pixels, schema=PIXEL_SCHEMA) \
        .write.format("delta").partitionBy("channel", "year").mode("overwrite") \
        .option("overwriteSchema", "true").save(table("pixels"))
    out["pixels_seconds"] = time.time() - t0
    out["pixels"] = table_stats(table("pixels"))
    out["times_table"] = table_stats(table("times"))
    out["files_table"] = table_stats(table("files"))
    out["grid_shape"] = [len(lat), len(lon)]
    spark.stop()
    return out


def silver() -> dict:
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    channels = [c for c, _, _ in oad_channels()]
    spark = _spark("datect-modis-silver")
    out = {}
    t0 = time.time()
    times = spark.read.format("delta").load(table("times"))
    first = Window.partitionBy("channel", "time").orderBy("file_rank", "time_pos")
    winners = (times.withColumn("rn", F.row_number().over(first))
               .withColumn("n_sources", F.count("*").over(Window.partitionBy("channel", "time")))
               .where("rn = 1").drop("rn"))
    winners.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(table("winners"))
    winners = spark.read.format("delta").load(table("winners"))
    cube_times = (winners.groupBy("time", "year").agg(F.countDistinct("channel").alias("n_channels"))
                  .where(F.col("n_channels") == len(channels)).drop("n_channels"))
    cube_times.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(table("cube_times"))
    cube_times = spark.read.format("delta").load(table("cube_times"))
    n_times = cube_times.count()
    w = winners.toPandas()
    out["unique_times_per_channel"] = w.groupby("channel").size().to_dict()
    out["duplicate_time_rows_dropped"] = int((w["n_sources"] - 1).sum())
    out["cube_times"] = int(n_times)
    out["times_seconds"] = time.time() - t0

    t0 = time.time()
    keep = (winners.join(cube_times.select("time"), "time")
            .select("channel", "time", "file_rank", "time_pos"))
    pixels = spark.read.format("delta").load(table("pixels"))
    chosen = pixels.join(F.broadcast(keep), ["channel", "time", "file_rank", "time_pos"])
    # Range-partitioning on time satisfies the aggregation's clustering, so this is
    # the only shuffle, and each output file holds a contiguous run of dates
    cube = (chosen.repartitionByRange(max(1, n_times // 12), "time")
            .groupBy("year", "time", "lat_idx", "lon_idx")
            .agg(*[F.max(F.when(F.col("channel") == c, F.col("value"))).alias(c) for c in channels])
            .withColumn("valid", F.expr(" AND ".join(f"{c} IS NOT NULL" for c in channels)))
            .sortWithinPartitions("time", "lat_idx", "lon_idx"))
    cube.write.format("delta").partitionBy("year").mode("overwrite") \
        .option("overwriteSchema", "true").save(table("cube"))
    out["cube_seconds"] = time.time() - t0
    out["cube"] = table_stats(table("cube"))
    out["valid_rows"] = spark.read.format("delta").load(table("cube")).where("valid").count()
    spark.stop()
    return out


def _region_counts(spark):
    """Valid pixels per (region, time); the scan is filtered to the regions' envelope
    so Delta can skip files by their lat_idx/lon_idx statistics."""
    from pyspark.sql import functions as F

    grid = spark.read.format("delta").load(table("grid")).toPandas()
    boxes = region_boxes(grid.loc[grid["axis"] == "lat", "coord"].to_numpy(),
                         grid.loc[grid["axis"] == "lon", "coord"].to_numpy())
    cube = (spark.read.format("delta").load(table("cube"))
            .where(F.col("valid"))
            .where(F.col("lat_idx").between(int(boxes["lat_lo"].min()), int(boxes["lat_hi"].max())))
            .where(F.col("lon_idx").between(int(boxes["lon_lo"].min()), int(boxes["lon_hi"].max())))
            .select("time", "lat_idx", "lon_idx"))
    b = F.broadcast(spark.createDataFrame(boxes))
    counts = (cube.join(b, F.col("lat_idx").between(b.lat_lo, b.lat_hi)
                        & F.col("lon_idx").between(b.lon_lo, b.lon_hi))
              .groupBy("region", "time").agg(F.count("*").alias("n_valid")))
    full = (spark.read.format("delta").load(table("cube_times")).select("time")
            .crossJoin(b.select("region_order", "region", "region_pixels"))
            .join(counts, ["region", "time"], "left")
            .fillna(0, subset=["n_valid"])
            .withColumn("valid_frac", F.col("n_valid").cast("double") / F.col("region_pixels")))
    return full, boxes


def _timed_count(spark) -> float:
    t0 = time.time()
    _region_counts(spark)[0].agg({"n_valid": "sum"}).collect()
    return time.time() - t0


def optimize() -> dict:
    from delta.tables import DeltaTable

    spark = _spark("datect-modis-optimize")
    out = {"max_file_bytes": OPTIMIZE_MAX_FILE_BYTES}

    # Bronze: compaction (one small file per source NetCDF -> a few per partition)
    before = table_stats(table("pixels"))
    t0 = time.time()
    DeltaTable.forPath(spark, table("pixels")).optimize().executeCompaction()
    out["bronze_pixels"] = {"before": before, "after": table_stats(table("pixels")),
                            "seconds": time.time() - t0}

    # Silver: Z-order on the grid indices, measured by the region aggregation
    grid = spark.read.format("delta").load(table("grid")).toPandas()
    boxes = region_boxes(grid.loc[grid["axis"] == "lat", "coord"].to_numpy(),
                         grid.loc[grid["axis"] == "lon", "coord"].to_numpy())

    def skipping():
        files = live_files(table("cube"))
        rows = {}
        for box in boxes.itertuples():
            n, size = files_overlapping(files, box)
            rows[box.region] = {"files_read": n, "bytes_read": size}
        return {"files": len(files), "bytes": int(files["size"].sum()), "per_region": rows}

    out["silver_cube"] = {"before": {**table_stats(table("cube")), "skipping": skipping(),
                                     "region_query_seconds": [_timed_count(spark), _timed_count(spark)]}}
    t0 = time.time()
    DeltaTable.forPath(spark, table("cube")).optimize().executeZOrderBy("lon_idx", "lat_idx")
    out["silver_cube"]["zorder_seconds"] = time.time() - t0
    out["silver_cube"]["after"] = {**table_stats(table("cube")), "skipping": skipping(),
                                   "region_query_seconds": [_timed_count(spark), _timed_count(spark)]}
    spark.stop()
    return out


def gold() -> dict:
    spark = _spark("datect-modis-gold")
    t0 = time.time()
    full, boxes = _region_counts(spark)
    full.select("time", "region_order", "region", "valid_frac", "n_valid", "region_pixels") \
        .withColumnRenamed("time", "date") \
        .write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(table("valid_frac"))
    out = {"seconds": time.time() - t0, "regions": boxes.to_dict("records"),
           "table": table_stats(table("valid_frac"))}
    spark.stop()
    return out


def parity() -> dict:
    import xarray as xr
    from pyspark.sql import functions as F

    spark = _spark("datect-modis-parity")
    out = {}

    # 1. gold == oad_cloud_fractions.parquet, exactly
    gold_df = spark.read.format("delta").load(table("valid_frac")).toPandas()
    gold_df["date"] = pd.to_datetime(gold_df["date"]).astype("datetime64[ns]")
    gold_df = gold_df.sort_values(["region_order", "date"]).reset_index(drop=True)
    ref = pd.read_parquet(REFERENCE_PARQUET)
    if YEARS:
        ref = ref[ref["date"].dt.year.astype(str).isin(YEARS)]
    ref = ref.reset_index(drop=True)
    mine = gold_df[ref.columns].astype(ref.dtypes.to_dict())
    merged = mine.merge(ref, on=["date", "region"], how="outer", suffixes=("", "_ref"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    out["valid_frac"] = {"rows": len(mine), "reference_rows": len(ref),
                         "unmatched_rows": int((merged["_merge"] != "both").sum()),
                         "max_abs_diff": float((both["valid_frac"] - both["valid_frac_ref"]).abs().max()),
                         "rows_not_identical": int((both["valid_frac"] != both["valid_frac_ref"]).sum())}
    try:
        pd.testing.assert_frame_equal(mine, ref, check_exact=True)
        out["valid_frac"]["assert_frame_equal_exact"] = True
    except AssertionError as e:
        out["valid_frac"]["assert_frame_equal_exact"] = False
        out["valid_frac"]["error"] = str(e)[:2000]

    # 2. time axis == cube.zarr
    cube = xr.open_zarr(CUBE_PATH)
    axis = pd.to_datetime(cube["time"].values)
    if YEARS:
        axis = axis[axis.year.astype(str).isin(YEARS)]
    axis = axis.values.astype("datetime64[ns]")
    mine_axis = pd.to_datetime(spark.read.format("delta").load(table("cube_times"))
                               .toPandas()["time"]).sort_values().values.astype("datetime64[ns]")
    out["time_axis"] = {"cube_zarr": len(axis), "silver": len(mine_axis),
                        "equal": bool(np.array_equal(axis, mine_axis))}

    # 3. five dates rebuilt from silver with the cube's stored stats == cube.zarr
    channels = oad_channels()
    picks = axis[np.linspace(0, len(axis) - 1, 5).astype(int)]
    n_lat, n_lon = cube.sizes["lat"], cube.sizes["lon"]
    cube_rows = (spark.read.format("delta").load(table("cube"))
                 .where(F.col("time").isin([pd.Timestamp(t).to_pydatetime() for t in picks])).toPandas())
    cube_rows["time"] = pd.to_datetime(cube_rows["time"]).astype("datetime64[ns]")
    dates = []
    for t in picks:
        rows = cube_rows[cube_rows["time"] == t]
        ref_data = cube["data"].sel(time=t).values
        ref_mask = cube["mask"].sel(time=t).values
        lat_idx, lon_idx = rows["lat_idx"].to_numpy(), rows["lon_idx"].to_numpy()
        mask = np.zeros((n_lat, n_lon), dtype=bool)
        mask[lat_idx, lon_idx] = rows["valid"].to_numpy()
        entry = {"date": str(pd.Timestamp(t).date()), "mask_equal": bool(np.array_equal(mask, ref_mask)),
                 "valid_pixels": int(mask.sum())}
        for i, (name, _, log) in enumerate(channels):
            raw = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
            present = rows[name].notna().to_numpy()
            raw[lat_idx[present], lon_idx[present]] = rows[name].to_numpy(np.float32)[present]
            got = standardize(raw, log, cube.attrs[f"{name}_mean"], cube.attrs[f"{name}_std"])
            want = ref_data[i]
            entry[name] = {"nan_equal": bool(np.array_equal(np.isnan(got), np.isnan(want))),
                           "bitwise_equal": bool(np.array_equal(got, want, equal_nan=True)),
                           "max_abs_diff": float(np.nanmax(np.abs(got - want))) if np.isfinite(want).any() else 0.0}
        dates.append(entry)
    out["five_dates"] = dates
    out["passed"] = bool(out["valid_frac"]["assert_frame_equal_exact"] and out["time_axis"]["equal"]
                         and all(d["mask_equal"] and all(d[c]["nan_equal"] and d[c]["max_abs_diff"] < 1e-5
                                                         for c, _, _ in channels) for d in dates))
    spark.stop()
    if not out["passed"]:
        print(json.dumps(out, indent=2, default=str))
        raise SystemExit("parity failed")
    return out


def vacuum() -> dict:
    """Drop the files OPTIMIZE replaced (the lake is scratch; history is not needed)."""
    from delta.tables import DeltaTable

    spark = _spark("datect-modis-vacuum")
    out = {}
    for key in ("pixels", "cube"):
        before = du_bytes(table(key))
        DeltaTable.forPath(spark, table(key)).vacuum(0)
        out[key] = {"bytes_on_disk_before": before, "bytes_on_disk_after": du_bytes(table(key))}
    out["lake_bytes_on_disk"] = du_bytes(lake_path())
    spark.stop()
    return out


def reference() -> dict:
    """Time the reference: 02_build_cube.py (cube written to scrubbed) and
    compute_region_valid_fractions, and check both against the committed outputs."""
    import xarray as xr

    ref_cube = os.environ.get("DATECT_REF_CUBE",
                              os.path.join(os.path.dirname(lake_path()), "reference", "cube.zarr"))
    # 02_build_cube.py reads oad/data/raw next to oad/config.py: point it at the data
    raw_link = Path(REPO_ROOT) / "oad" / "data" / "raw"
    raw_link.parent.mkdir(parents=True, exist_ok=True)
    if not raw_link.exists():
        raw_link.symlink_to(RAW_ROOT)
    out = {"reference_cube": ref_cube}
    t0 = time.time()
    subprocess.run([sys.executable, "oad/scripts/02_build_cube.py", "--out", ref_cube],
                   cwd=REPO_ROOT, check=True)
    out["build_cube_seconds"] = time.time() - t0
    out["build_cube_max_rss_gb"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6

    confound = _load_oad_module("oad_cloud_confound", os.path.join("scripts", "07_cloud_confound.py"))
    t0 = time.time()
    fracs = confound.compute_region_valid_fractions(Path(ref_cube))
    out["valid_fractions_seconds"] = time.time() - t0
    out["total_seconds"] = out["build_cube_seconds"] + out["valid_fractions_seconds"]
    fracs["date"] = pd.to_datetime(fracs["date"])
    committed = pd.read_parquet(REFERENCE_PARQUET)
    out["rebuilt_equals_committed_parquet"] = bool(fracs.reset_index(drop=True).equals(committed.reset_index(drop=True)))
    original = xr.open_zarr(CUBE_PATH)
    rebuilt = xr.open_zarr(ref_cube)
    out["rebuilt_time_axis_equal"] = bool(np.array_equal(original["time"].values, rebuilt["time"].values))
    out["rebuilt_mask_equal"] = bool((original["mask"] == rebuilt["mask"]).all().compute())
    out["reference_cube_bytes_on_disk"] = du_bytes(ref_cube)
    return out


STEPS = {"bronze": bronze, "silver": silver, "optimize": optimize, "gold": gold,
         "parity": parity, "vacuum": vacuum, "reference": reference}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("step", choices=list(STEPS))
    parser.add_argument("--force", action="store_true", help="re-run even if this commit already ran it")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    out_file = os.path.join(OUT_DIR, f"modis_{args.step}.json")
    sha = _git_sha()
    if not args.force and os.path.exists(out_file):
        with open(out_file) as f:
            if json.load(f).get("git_sha") == sha:
                print(f"{args.step}: already done at {sha[:8]}, skipping")
                return
    t0 = time.time()
    result = STEPS[args.step]()
    result = {"step": args.step, "git_sha": sha, "host": socket.gethostname(),
              "cpus": os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count())),
              "wall_seconds": time.time() - t0, **result}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    mlflow = get_mlflow("datect-modis-lake")
    with mlflow.start_run(run_name=args.step):
        mlflow.log_params({"step": args.step, "git_sha": sha, "raw_root": RAW_ROOT})
        flat = pd.json_normalize(result, sep=".").iloc[0]
        # MLflow keys allow only [A-Za-z0-9_-./ ]; region names have dashes and parentheses
        mlflow.log_metrics({re.sub(r"[^A-Za-z0-9_\-./ ]", "_", k): float(v) for k, v in flat.items()
                            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)})
        mlflow.log_artifact(out_file)
    print(f"{args.step}: done in {result['wall_seconds']:.0f}s -> {out_file}")


if __name__ == "__main__":
    main()

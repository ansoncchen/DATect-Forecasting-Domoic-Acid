"""MODIS -> Delta pipeline (datect_spark/modis_lake.py): the parts that run without
Spark. The Spark steps and the exact parity checks run on Hyak."""
import os
import time

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")


@pytest.fixture(scope="module")
def lake():
    # datect_spark.session pins TZ=UTC at import; give the rest of the suite its zone back
    saved = os.environ.get("TZ")
    from datect_spark import modis_lake
    if saved is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = saved
    time.tzset()
    return modis_lake


LAT = np.linspace(41.0, 49.0, 9)
LON = np.linspace(-131.0, -120.8, 7)


def _write_nc(path, times, values):
    ds = xr.Dataset(
        {"chlorophyll": (("time", "altitude", "latitude", "longitude"), values[:, None].astype(np.float32))},
        coords={"time": pd.to_datetime(times), "altitude": [0.0], "latitude": LAT, "longitude": LON})
    ds.to_netcdf(path, engine="netcdf4", encoding={"chlorophyll": {"_FillValue": np.float32(-9999999.0)}})


@pytest.fixture
def raw_dir(tmp_path):
    for ch in ("chla", "k490", "nflh", "sst"):
        (tmp_path / ch).mkdir()
    rng = np.random.default_rng(0)
    monthly = rng.random((3, len(LAT), len(LON)))
    monthly[0, :2] = np.nan       # partly cloudy
    monthly[2] = np.nan           # fully cloudy: no pixels, but the time still exists
    _write_nc(tmp_path / "chla" / "chla_2003_01.nc", ["2003-01-05", "2003-01-13", "2003-01-21"], monthly)
    _write_nc(tmp_path / "chla" / "chla_20030105.nc", ["2003-01-05"], monthly[:1])
    (tmp_path / "chla" / "._chla_2003_01.nc").write_bytes(b"\0" * 163)
    return tmp_path, monthly


def test_files_ranked_in_reference_order_and_stubs_skipped(lake, raw_dir):
    root, _ = raw_dir
    files, stubs = lake.list_files(str(root))
    # sorted(glob("chla_*.nc")): per-day files sort before the monthly file of
    # their year, so drop_duplicates(keep="first") keeps the per-day copy
    assert [(r[2], r[4]) for r in files] == [("chla_20030105.nc", 0), ("chla_2003_01.nc", 1)]
    assert [os.path.basename(s) for s in stubs] == ["._chla_2003_01.nc"]


def test_file_times_and_pixels_match_the_netcdf(lake, raw_dir):
    root, monthly = raw_dir
    files, _ = lake.list_files(str(root))
    pdf = pd.DataFrame(files, columns=lake.FILE_COLUMNS)
    monthly_file = pdf[pdf["file_name"] == "chla_2003_01.nc"]

    times = pd.concat(lake._file_times(iter([monthly_file])))
    assert list(times["time"]) == list(pd.to_datetime(["2003-01-05", "2003-01-13", "2003-01-21"]))
    assert list(times["time_pos"]) == [0, 1, 2]
    assert times["grid_sha"].nunique() == 1 and times["n_lat"].iloc[0] == len(LAT)

    pixels = pd.concat(lake._file_pixels(iter([monthly_file])))
    rebuilt = np.full(monthly.shape, np.nan, dtype=np.float32)
    rebuilt[pixels["time_pos"], pixels["lat_idx"], pixels["lon_idx"]] = pixels["value"]
    np.testing.assert_array_equal(rebuilt, monthly.astype(np.float32))
    assert pixels["value"].notna().all()  # NaN (decoded fill) pixels are not stored
    assert set(pixels["time_pos"]) == {0, 1}  # the fully cloudy frame has no rows


def test_region_boxes_match_the_reference_pixel_count(lake):
    lat = np.round(np.arange(321) * 0.025 + 41.0, 3)
    lon = np.round(np.arange(409) * 0.025 - 131.0, 3)
    boxes = lake.region_boxes(lat, lon)
    for box, region in zip(boxes.itertuples(), lake.oad_regions()):
        lat_in = (lat >= region.lat_min) & (lat <= region.lat_max)
        lon_in = (lon >= region.lon_min) & (lon <= region.lon_max)
        assert box.region == region.name
        assert box.region_pixels == int(lat_in.sum() * lon_in.sum())
        assert box.lat_hi - box.lat_lo + 1 == lat_in.sum()


def test_standardize_reproduces_the_cube_transform_bitwise(lake):
    rng = np.random.default_rng(1)
    raw = rng.lognormal(size=(5, 6)).astype(np.float32)
    raw[0, 0] = np.nan
    raw[1, 1] = 1e-6  # below the log clip
    da = xr.DataArray(raw)
    # 02_build_cube.py: np.log(da.clip(min=LOG_CLIP_MIN)), then (da - mu) / (sigma + 1e-8)
    logged = np.log(da.clip(min=1e-4))
    mu, sigma = float(np.nanmean(logged.values)), float(np.nanstd(logged.values))
    want = ((logged - mu) / (sigma + 1e-8)).values
    got = lake.standardize(raw, True, mu, sigma)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, want)


def test_live_files_replays_the_delta_log(lake, tmp_path):
    deltalake = pytest.importorskip("deltalake")
    path = str(tmp_path / "t")
    df = pd.DataFrame({"lat_idx": np.array([3, 9], dtype=np.int16), "lon_idx": np.array([1, 2], dtype=np.int16)})
    deltalake.write_deltalake(path, df)
    deltalake.write_deltalake(path, df.assign(lat_idx=np.array([5, 7], dtype=np.int16)), mode="overwrite")
    files = lake.live_files(path)
    assert len(files) == 1 and files["num_records"].iloc[0] == 2
    assert (files["min_lat_idx"].iloc[0], files["max_lat_idx"].iloc[0]) == (5, 7)
    box = pd.DataFrame([{"lat_lo": 8, "lat_hi": 20, "lon_lo": 0, "lon_hi": 5}]).iloc[0]
    assert lake.files_overlapping(files, box) == (0, 0)

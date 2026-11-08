"""
DATect Web Application API
FastAPI backend providing forecasting, visualization, and analysis endpoints
"""

import logging
import os
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Literal, Optional
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


from forecasting import ForecastEngine
import config
from backend import band
from backend.band import forecast_band
from forecasting.metrics import alert_parts
from forecasting.raw_data_forecaster import load_raw_da_measurements
from forecasting.tuned_config import get_eval_windows
from backend.visualizations import (
    generate_correlation_heatmap,
    generate_sensitivity_analysis,
    generate_time_series_comparison,
    generate_waterfall_plot,
    generate_spectral_analysis,
    generate_band_plot,
)
from backend import precomputed, track_record
from backend.records import clean_for_json, retrospective_records  # noqa: F401 (re-exported)

# Configure logging
logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isabs(config.FINAL_OUTPUT_PATH):
    config.FINAL_OUTPUT_PATH = os.path.join(project_root, config.FINAL_OUTPUT_PATH)


def _resolve_repo_relative_path(rel_path: str) -> str:
    p = (rel_path or "").strip()
    if os.path.isabs(p):
        return p
    return os.path.join(project_root, p.lstrip("./"))


app = FastAPI(
    title="DATect API",
    description="Domoic Acid Forecasting System REST API",
    version="1.0.0"
)

# CORS middleware (configurable for production)
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    # Local dev only — set ALLOWED_ORIGINS for production (comma-separated)
    origins = ["http://localhost:3000", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False if "*" in origins else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Lazy singletons for Cloud Run optimization
forecast_engine = None
# Precomputed store (Delta), loaded at startup when DATECT_PRECOMPUTED_DIR is set
precomputed.load_from_env()

@lru_cache(maxsize=1)
def load_data_cached() -> pd.DataFrame:
    """Load final output data once per process for reuse."""
    return pd.read_parquet(config.FINAL_OUTPUT_PATH)

def get_data_copy() -> pd.DataFrame:
    """Return a safe copy of cached data for mutation."""
    return load_data_cached().copy()

def get_forecast_engine() -> ForecastEngine:
    global forecast_engine
    if forecast_engine is None:
        # Skip validation on init for faster startup
        forecast_engine = ForecastEngine(validate_on_init=False)
    return forecast_engine

def _single_forecast(site: str, forecast_date, task: str, model_type: str):
    """(generate_single_forecast output, model_version): from the precomputed store
    when it holds this request for the same code, config and data
    (DATECT_PRECOMPUTED_DIR), computed on demand otherwise (model_version None)."""
    hit = precomputed.forecast(site, forecast_date, task, model_type)
    if hit is not None:
        return hit.result, hit.model_version
    result = get_forecast_engine().generate_single_forecast(
        config.FINAL_OUTPUT_PATH, forecast_date, site, task, model_type
    )
    return result, None

@lru_cache(maxsize=1)
def get_site_mapping() -> dict:
    """Get cached site name mapping for flexible API access"""
    data = load_data_cached()
    return {s.lower().replace(' ', '-'): s for s in data['site'].unique()}

def resolve_site(site: str) -> str:
    """Resolve site name to canonical form using cached mapping"""
    mapping = get_site_mapping()
    return mapping.get(site.lower(), site)


# Pydantic models
class ForecastRequest(BaseModel):
    date: date
    site: str
    task: str = "regression"  # "regression" or "classification"
    model: str = "ensemble"

class SiteInfo(BaseModel):
    sites: List[str]
    date_range: Dict[str, date]
    site_mapping: Dict[str, str]

@app.get("/api")
def root():
    """Root endpoint with API information."""
    return {
        "message": "DATect API - Domoic Acid Forecasting System",
        "version": "1.0.0",
            "docs": "/docs"
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat(),
            "precomputed": precomputed.status()}

@app.get("/api/sites", response_model=SiteInfo)
def get_sites():
    """Get available sites and date range from the dataset."""
    try:
        # Load data to get site information
        data = get_data_copy()
        data['date'] = pd.to_datetime(data['date'])
        
        sites = sorted(data['site'].unique().tolist())
        # Also provide lowercase versions for easier API access
        site_mapping = {site.lower().replace(' ', '-'): site for site in sites}
        date_range = {
            "min": data['date'].min().date(),
            "max": data['date'].max().date()
        }
        
        return SiteInfo(sites=sites, date_range=date_range, site_mapping=site_mapping)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load site information: {str(e)}")

@app.get("/api/historical/{site}")
def get_historical_data(
    site: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = 1000
):
    """Real DA measurements at a site, one per site-day at the max: not the
    gap-filled weekly series the models train on."""
    actual_site = _known_site(site)
    m = _raw_measurements()
    rows = m[m["site"] == actual_site]
    if start_date:
        rows = rows[rows["date"] >= _request_date(start_date)]
    if end_date:
        rows = rows[rows["date"] <= _request_date(end_date)]
    rows = rows.sort_values("date").tail(limit)
    return {
        "site": actual_site,
        "series": "raw measurements, one per site-day at the max",
        "count": len(rows),
        "rows": [{"date": d.date().isoformat(), "value": float(v)} for d, v in zip(rows["date"], rows["da_raw"])],
    }


@lru_cache(maxsize=1)
def _load_metrics_summary() -> dict:
    """Headline evaluation metrics, read from the committed results in results/.

    The dashboard shows these instead of hardcoding numbers that drift from the
    latest evaluation: the deterministic chronological holdout (overall and per
    site) and the 5-seed bootstrap confirmation on the same window.
    """
    import glob
    import json

    from sklearn.metrics import r2_score

    from forecasting.tuned_config import get_eval_windows

    window = get_eval_windows()
    start, end = window["holdout_start"], window["holdout_end"]
    tag = f"{start.replace('-', '')}_{end.replace('-', '')}"
    results = Path(project_root) / "results"
    chrono = json.loads(
        (results / "chronological" / f"chronological_regression_ensemble_{tag}.json").read_text()
    )

    seed_r2 = []
    for path in sorted(glob.glob(str(results / "multi_seed" / "*_predictions.parquet"))):
        df = pd.read_parquet(path, columns=["date", "actual_da", "predicted_da"])
        df["date"] = pd.to_datetime(df["date"])
        held = df[(df["date"] >= start) & (df["date"] < end)]
        seed_r2.append(float(r2_score(held["actual_da"], held["predicted_da"])))

    reliability = {}
    for name in ("holdout", "validation"):
        sites = sorted(track_record.window_rows(name)["site"].unique())
        reliability[name] = {
            "summary": track_record.window_summary(name),
            "per_site": {site: track_record.window_summary(name, site) for site in sites},
        }

    return {
        "holdout_window": {"start": start, "end": end},
        "overall": chrono["overall"],
        "per_site": chrono["per_site"],
        "multi_seed": {
            "n_seeds": len(seed_r2),
            "r2_mean": float(np.mean(seed_r2)) if seed_r2 else None,
            "r2_std": float(np.std(seed_r2)) if seed_r2 else None,
        },
        "reliability": reliability,
        "source": {
            "chronological": f"results/chronological/chronological_regression_ensemble_{tag}.json",
            "band": track_record.BAND_PREDICTIONS_PATH,
            "multi_seed": "results/multi_seed/*_predictions.parquet",
        },
    }


EARLIEST_DATE, LATEST_DATE = pd.Timestamp("1900-01-01"), pd.Timestamp("2200-01-01")


def _request_date(day) -> pd.Timestamp:
    """A request's date as a Timestamp, or 422 outside the supported range. Pandas
    cannot represent dates far from the present, and nothing here is near them."""
    try:
        when = pd.Timestamp(day)
    except (ValueError, OverflowError) as e:
        raise HTTPException(status_code=422, detail=f"Date out of range: {day}") from e
    if not EARLIEST_DATE <= when <= LATEST_DATE:
        raise HTTPException(status_code=422, detail=f"Date out of range: {day}")
    return when


def _known_site(site: str) -> str:
    """Canonical site name, or 404 for a site DATect does not forecast."""
    resolved = resolve_site(site)
    if resolved not in config.SITES:
        raise HTTPException(status_code=404, detail=f"Unknown site: {site}")
    return resolved


@app.get("/api/track-record")
def get_track_record(window: track_record.Window = "holdout", site: str | None = None):
    """Chronological forecasts for a window with every summary computed here:
    overall (or one site's), per site, and the rows behind them."""
    site = _known_site(site) if site is not None else None
    start, end = track_record.window_bounds(window)
    rows = track_record.window_rows(window, site)
    per_site = {s: track_record.window_summary(window, s) for s in sorted(rows["site"].unique())}
    return clean_for_json({
        "window": {"name": window, "start": start, "end": end},
        "summary": track_record.window_summary(window, site),
        "per_site": per_site,
        "rows": track_record.records(rows),
    })


@app.get("/api/sites/{site}/history")
def get_site_history(site: str, since: date = date(2019, 1, 1), until: date | None = None):
    """Each real measurement at a site with the forecast made for it, its band and alert."""
    site = _known_site(site)
    _request_date(since)
    if until:
        _request_date(until)
    rows = track_record.site_history(site, since.isoformat(), until.isoformat() if until else None)
    return clean_for_json({"site": site, "rows": rows})


@app.get("/api/metrics/summary")
def get_metrics_summary():
    """Headline holdout metrics (overall, per site, multi-seed) from results/."""
    try:
        return clean_for_json(_load_metrics_summary())
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Evaluation results not found: {e}") from e


@app.get("/api/config")
def get_config():
    """Get current system configuration."""
    return {
        "forecast_mode": config.FORECAST_MODE,
        "forecast_task": config.FORECAST_TASK,
        "forecast_model": config.FORECAST_MODEL,
        "forecast_horizon_weeks": config.FORECAST_HORIZON_WEEKS,
        "forecast_horizon_days": config.FORECAST_HORIZON_DAYS,
        "spike_threshold": config.SPIKE_THRESHOLD,
        "regression_alert_threshold": config.SPIKE_REGRESSION_ALERT_THRESHOLD,
        "classifier_alert_cutoff": config.SPIKE_ALERT_PROB_THRESHOLD,
        "alert_levels": _alert_levels(),
        "band_level": band.chosen_variant()["coverage"],
        "category_bins": [0, *config.DA_CATEGORY_BINS[1:-1], None],
        "category_labels": CATEGORY_LABELS,
        "eval_windows": get_eval_windows(),
        "sites_north_to_south": _sites_north_to_south(),
        "newest_measurement_date": _raw_measurements()["date"].max().date().isoformat(),
        "latest_forecast_week": _latest_forecast_week().date().isoformat(),
    }


def _sites_north_to_south() -> list:
    return sorted(config.SITES, key=lambda site: -config.SITES[site][0])


def _latest_forecast_week() -> pd.Timestamp:
    """The newest weekly row: the dashboard's default date, always precomputed."""
    return pd.to_datetime(load_data_cached()["date"]).max()


def _alert_levels() -> list:
    """The dashboard's alert levels, each with its rule and cutoff."""
    p, r, w = (config.SPIKE_ALERT_PROB_THRESHOLD, config.SPIKE_REGRESSION_ALERT_THRESHOLD,
               config.ALERT_WARNING_PROB_CUTOFF)
    return [
        {"name": "Quiet", "rule": "no alert condition holds", "cutoff": None, "placeholder": False},
        {"name": "Watch", "rule": f"classifier probability >= {p:g} or forecast >= {r:g} µg/g",
         "cutoff": p, "placeholder": False},
        {"name": "Warning", "rule": f"classifier probability >= {w:g}", "cutoff": w,
         "placeholder": False},
    ]

# Visualization endpoints
# NOTE: More specific routes must come before generic parameter routes
@app.get("/api/visualizations/correlation/all")
def get_correlation_heatmap_all():
    """Generate correlation heatmap for all sites combined."""
    try:
        data = get_data_copy()
        plot_data = generate_correlation_heatmap(data, site=None)
        return {"success": True, "plot": plot_data, "cached": False, "source": "computed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate correlation heatmap: {str(e)}")

@app.get("/api/visualizations/correlation/{site}")
def get_correlation_heatmap_single(site: str):
    """Generate correlation heatmap for a single site."""
    try:
        data = get_data_copy()
        actual_site = resolve_site(site)

        cached_plot = precomputed.correlation(actual_site)
        if cached_plot is not None:
            return {"success": True, "plot": cached_plot, "cached": True, "source": "precomputed"}

        plot_data = generate_correlation_heatmap(data, actual_site)
        return {"success": True, "plot": plot_data, "cached": False, "source": "computed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate correlation heatmap: {str(e)}")

@app.get("/api/visualizations/sensitivity/all")
def get_sensitivity_analysis_all():
    """Generate sensitivity analysis plots for all sites combined."""
    try:
        data = get_data_copy()
        plots = generate_sensitivity_analysis(data, site=None)
        return {"success": True, "plots": plots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate sensitivity analysis: {str(e)}")

@app.get("/api/visualizations/sensitivity/{site}")
def get_sensitivity_analysis_single(site: str):
    """Generate sensitivity analysis plots for a single site."""
    try:
        data = get_data_copy()
        actual_site = resolve_site(site)
        
        plots = generate_sensitivity_analysis(data, actual_site)
        return {"success": True, "plots": plots}
    except Exception as e:
        logging.error(f"Error in sensitivity analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate sensitivity analysis: {str(e)}")

@app.get("/api/visualizations/comparison/{site}")
def get_time_series_comparison_single(site: str):
    """Generate time series comparison for a single site."""
    try:
        data = get_data_copy()
        actual_site = resolve_site(site)
        
        plot_data = generate_time_series_comparison(data, actual_site)
        return {"success": True, "plot": plot_data}  # plot_data is already in correct format
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate time series comparison: {str(e)}")

@app.get("/api/visualizations/waterfall")
def get_waterfall_plot():
    """Generate waterfall plot for all sites."""
    try:
        data = get_data_copy()
        plot_data = generate_waterfall_plot(data)
        return {"success": True, "plot": plot_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate waterfall plot: {str(e)}")

@app.get("/api/visualizations/spectral/all")
def get_spectral_analysis_all():
    """Spectral analysis for all sites combined (from the precomputed store when it has it)."""
    try:
        plots = precomputed.spectral(None)
        
        if plots is not None:
            return {"success": True, "plots": plots, "cached": True, "source": "precomputed"}

        # Compute on server (expensive - only for local development)
        logging.warning("Computing spectral analysis on server - this is very expensive!")
        data = get_data_copy()
        plots = generate_spectral_analysis(data, site=None, **_spectral_model_results())
        return {"success": True, "plots": plots, "cached": False, "source": "computed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate spectral analysis: {str(e)}")

@app.get("/api/visualizations/spectral/{site}")
def get_spectral_analysis_single(site: str):
    """Spectral analysis for a single site (from the precomputed store when it has it)."""
    try:
        data = get_data_copy()
        actual_site = resolve_site(site)

        plots = precomputed.spectral(actual_site)
        
        if plots is not None:
            return {"success": True, "plots": plots, "cached": True, "source": "precomputed"}

        # Compute on server (expensive - only for local development)
        logging.warning(f"Computing spectral analysis for {actual_site} on server - this is very expensive!")
        plots = generate_spectral_analysis(data, actual_site, **_spectral_model_results())
        return {"success": True, "plots": plots, "cached": False, "source": "computed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate spectral analysis: {str(e)}")

def _spectral_model_results() -> dict:
    """The XGBoost and Random Forest retrospective predictions the spectral analysis
    compares with, from the precomputed store (None for a slice it lacks)."""
    out = {}
    for key, model in (("xgb_results", "xgboost"), ("rf_results", "rf")):
        records = precomputed.retrospective("regression", model)
        out[key] = pd.DataFrame(records) if records else None
    return out

def get_latest_da_from_raw_files():
    """Latest DA from raw CSVs; uses ``config.ORIGINAL_DA_FILES`` (same paths as the forecast pipeline)."""
    from forecasting.raw_data_forecaster import _normalize_site_name

    latest_da_data = {}

    for site_key, rel_path in config.ORIGINAL_DA_FILES.items():
        file_path = _resolve_repo_relative_path(rel_path)
        site_name = _normalize_site_name(site_key)

        if not os.path.exists(file_path):
            continue

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.warning("Failed to read raw DA file %s: %s", file_path, e)
            continue

        if df.empty:
            continue

        last_row = df.iloc[-1]
        date_obj = None
        da_value = None

        if "CollectDate" in df.columns and "Domoic Result" in df.columns:
            date_obj = pd.to_datetime(str(last_row["CollectDate"]), errors="coerce")
            da_value = last_row["Domoic Result"]
        elif (
            "Harvest Month" in df.columns
            and "Harvest Date" in df.columns
            and "Harvest Year" in df.columns
            and "Domoic Acid" in df.columns
        ):
            date_str = (
                f"{last_row['Harvest Month']} {last_row['Harvest Date']}, "
                f"{last_row['Harvest Year']}"
            )
            date_obj = pd.to_datetime(date_str, errors="coerce")
            da_value = last_row["Domoic Acid"]
        else:
            logger.warning("Unknown raw DA schema for %s", file_path)
            continue

        if pd.isna(date_obj):
            logger.warning("Unparseable date in last row of %s", file_path)
            continue

        from forecasting.measurements import parse_da_values

        da_numeric = float(parse_da_values([da_value]).iloc[0])  # "<1" -> LOD/2
        if pd.isna(da_numeric):
            logger.warning("Unparseable DA value in last row of %s", file_path)
            continue

        latest_da_data[site_name] = {"da": da_numeric, "date": date_obj}

    return latest_da_data

MAP_CATEGORY_COLORS = ("#9aa5b1", "#e0a100", "#e4670f", "#c0392b")  # by category: gray, never green


@app.get("/api/visualizations/map")
def get_site_map():
    """Map of the monitoring sites coloured by the last measured DA category. It reports
    a reading that may be months old, so it gives no risk verdict and no reassuring green."""
    try:
        # Import here to avoid issues if plotly is not available during startup
        import plotly.graph_objs as go
        
        # Get site coordinates from config and latest DA data from raw files
        sites = config.SITES
        latest_da_data = get_latest_da_from_raw_files()
        
        # Prepare data for map
        site_names = []
        latitudes = []
        longitudes = []
        colors = []
        hover_texts = []
        
        for site_name, (lat, lon) in sites.items():
            # Get latest DA data from raw files
            if site_name in latest_da_data:
                site_info = latest_da_data[site_name]
                recent_da = site_info['da']
                recent_date = site_info['date']
                
                category = int(np.searchsorted(config.DA_CATEGORY_BINS[1:-1], recent_da, side="right"))
                color = MAP_CATEGORY_COLORS[category]
                
                hover_text = f'<b>{site_name}</b><br>' + \
                            f'Latitude: {lat:.4f}<br>' + \
                            f'Longitude: {lon:.4f}<br>' + \
                            f'Last measured DA: {recent_da:.2f} µg/g ({CATEGORY_LABELS[category]})<br>' + \
                            f'Measured: {recent_date.strftime("%Y-%m-%d") if hasattr(recent_date, "strftime") else str(recent_date)}<extra></extra>'
            else:
                # No data available - use default color
                color = "#d0d5db"
                hover_text = f'<b>{site_name}</b><br>' + \
                            f'Latitude: {lat:.4f}<br>' + \
                            f'Longitude: {lon:.4f}<br>' + \
                            'Last measured DA: no data<extra></extra>'
            
            site_names.append(site_name)
            latitudes.append(lat)
            longitudes.append(lon)
            colors.append(color)
            hover_texts.append(hover_text)
        
        # Create map trace
        map_trace = go.Scattermapbox(
            lat=latitudes,
            lon=longitudes,
            mode='markers',
            marker=dict(
                size=14,
                color=colors,
                symbol='circle'
            ),
            text=site_names,
            textposition="top center",
            hovertemplate=hover_texts,
            name='Monitoring Sites'
        )
        
        # Calculate map center (average of all coordinates)
        center_lat = sum(latitudes) / len(latitudes)
        center_lon = sum(longitudes) / len(longitudes)
        
        # Create layout
        layout = go.Layout(
            title='DATect Monitoring Sites - Last Measured DA',
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=center_lat, lon=center_lon),
                zoom=5.5
            ),
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        # Create figure
        fig = go.Figure(data=[map_trace], layout=layout)
        
        # Convert to JSON format expected by frontend
        plot_json = fig.to_dict()
        
        return {"success": True, "plot": plot_json}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate site map: {str(e)}")



CATEGORY_LABELS = ['Low (<5)', 'Moderate (5–<20)', 'High (20–<40)', 'Extreme (≥40)']


@lru_cache(maxsize=1)
def _raw_measurements() -> pd.DataFrame:
    """Real DA measurements, one row per site-day at the max (``load_raw_da_measurements``)."""
    return load_raw_da_measurements()


def _recent_measurements(site: str, anchor_date, n: int = 6) -> list:
    """The last ``n`` real measurements at ``site`` dated on or before the anchor, oldest first."""
    m = _raw_measurements()
    seen = m[(m["site"] == site) & (m["date"] <= pd.Timestamp(anchor_date))].sort_values("date").tail(n)
    return [{"date": d.date().isoformat(), "value": float(v)} for d, v in zip(seen["date"], seen["da_raw"])]


def _no_forecast(site: str, forecast_date: pd.Timestamp, error: str, source: dict | None = None) -> dict:
    """A forecast that could not be made, in the same shape as one that was, with the
    measurements the forecast would have seen."""
    anchor = forecast_date - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
    recent = _recent_measurements(site, anchor)
    return {
        "success": False, "forecast_date": forecast_date.strftime('%Y-%m-%d'), "site": site,
        "anchor_date": anchor.date().isoformat(), "error": error,
        "band": None, "band_reason": None, "band_p_exceed": None, "category_probabilities": None,
        "alert": None, "last_measurement": recent[-1] if recent else None,
        "recent_measurements": recent, "graphs": {},
        **(source or {"source": "computed", "model_version": None}),
    }


def _forecast_payload(site: str, forecast_date: pd.Timestamp, model_type: str) -> dict:
    """One week's regression forecast with its conformal band, the alert's parts and
    the measurements the forecast could see (dashboard design, section 6)."""
    result, version = _single_forecast(site, forecast_date, "regression", model_type)
    source = {"source": "precomputed" if version is not None else "computed", "model_version": version}
    if result is None or "predicted_da" not in result:
        anchor = forecast_date - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
        latest = _latest_forecast_week()
        if forecast_date > latest:
            error = f"No forecast: {forecast_date.date().isoformat()} is past the newest data ({latest.date().isoformat()})"
        else:
            why = ("too little training data" if _recent_measurements(site, anchor)
                   else "no measurement at this site")
            error = f"No forecast: {why} before {anchor.date().isoformat()}"
        return _no_forecast(site, forecast_date, error, source)

    predicted = float(result["predicted_da"])
    anchor = pd.Timestamp(result["anchor_date"])
    recent = _recent_measurements(site, anchor)
    last = recent[-1] if recent else None
    if model_type == "ensemble":
        band_info = forecast_band(predicted, anchor)
        band_reason = None if band_info is not None else "fewer than 30 earlier forecast errors"
    else:
        band_info, band_reason = None, "calibrated for the ensemble only"
    alert = alert_parts(
        result.get("spike_probability"), float(result.get("ensemble_prediction", predicted)),
        last["value"] if last else None,
        prob_cutoff=config.SPIKE_ALERT_PROB_THRESHOLD,
        regression_cutoff=config.SPIKE_REGRESSION_ALERT_THRESHOLD,
        warning_cutoff=config.ALERT_WARNING_PROB_CUTOFF, threshold=config.SPIKE_THRESHOLD,
    )

    graphs = {}
    if band_info is not None:
        xgb_pred, rf_pred = result.get("xgb_prediction"), result.get("rf_prediction")
        graphs["level_range"] = {
            "type": "gradient_uncertainty",
            "gradient_plot": generate_band_plot(
                band_info, predicted,
                xgb_prediction=float(xgb_pred) if xgb_pred is not None else None,
                rf_prediction=float(rf_pred) if rf_pred is not None else None,
                threshold=config.SPIKE_THRESHOLD),
            "band": band_info,
        }
        graphs["category_range"] = {
            "type": "category_range", "source": "range",
            "predicted_category": int(np.searchsorted(config.DA_CATEGORY_BINS[1:-1], predicted, side="right")),
            "class_probabilities": band_info["category_probabilities"],
            "category_labels": CATEGORY_LABELS,
        }

    ensemble = {k: result[k] for k in ("naive_prediction", "xgb_prediction", "rf_prediction",
                                       "ensemble_prediction", "ensemble_weights") if k in result}
    return {
        "success": True,
        "forecast_date": forecast_date.strftime('%Y-%m-%d'),
        "site": site,
        "anchor_date": anchor.date().isoformat(),
        "regression": result,
        "ensemble": ensemble,
        "band": band_info,
        "band_reason": band_reason,
        "band_p_exceed": band_info["p_exceed"] if band_info else None,
        "category_probabilities": band_info["category_probabilities"] if band_info else None,
        "alert": alert,
        "last_measurement": last,
        "recent_measurements": recent,
        "graphs": graphs,
        **source,
    }


@app.post("/api/forecast/enhanced")
def generate_enhanced_forecast(request: ForecastRequest):
    """The dashboard's forecast: regression only (categories come from the band)."""
    try:
        site = resolve_site(request.site)
        model_type = getattr(request, 'model', None) or config.FORECAST_MODEL or "ensemble"
        return clean_for_json(_forecast_payload(site, _request_date(request.date), model_type))
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "forecast_date": str(request.date), "site": request.site, "error": str(e)}


@app.get("/api/forecast/all")
def get_all_sites_forecast(
    date: date | None = None,
    model: Literal["ensemble", "xgboost", "rf", "naive", "linear"] = "ensemble",
):
    """One week's forecast for every site, north to south (default: the latest week).
    A site whose forecast fails is reported with its error; the others still show."""
    day = _request_date(date) if date is not None else _latest_forecast_week()
    sites = []
    for site in _sites_north_to_south():
        try:
            payload = _forecast_payload(site, day, model)
        except Exception as e:
            logger.warning("Forecast for %s on %s failed: %s", site, day.date(), e)
            payload = _no_forecast(site, day, f"Forecast failed: {e}")
        ok = payload["success"]
        sites.append({
            "site": site,
            "anchor_date": payload.get("anchor_date"),
            "predicted_da": payload["regression"]["predicted_da"] if ok else None,
            "band": payload.get("band"),
            "alert": payload.get("alert"),
            "last_measurement": payload.get("last_measurement"),
            "source": payload["source"],
            "error": payload.get("error"),
        })
    return clean_for_json({"date": day.date().isoformat(), "sites": sites})


# Serve built frontend if present (single-origin deploy)
try:
    frontend_dist = os.path.join(project_root, "frontend", "dist")
    if os.path.isdir(frontend_dist):
        logging.info("Frontend dist found, mounting static files")
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")
    else:
        logging.warning(f"Frontend dist not found at: {frontend_dist}")
except Exception as e:
    logging.error(f"Error setting up static files: {e}")
    pass

def run_server(host: str = "0.0.0.0", port: int = None, use_granian: bool = None):
    """
    Run the API server with the best available ASGI server.
    
    Priority: Granian (fastest) > Uvicorn (standard)
    
    Args:
        host: Host to bind to
        port: Port to bind to (default: PORT env var or 8000)
        use_granian: Force Granian (True) or Uvicorn (False), auto-detect if None
    """
    if port is None:
        port = int(os.getenv("PORT", "8000"))
    
    # Auto-detect best server
    if use_granian is None:
        use_granian = os.getenv("DATECT_USE_GRANIAN", "1").lower() in ("1", "true", "yes")
    
    if use_granian:
        try:
            from granian import Granian
            from granian.constants import Interfaces
            
            logging.info(f"Starting Granian server (Rust-based, 20-40% faster) on {host}:{port}")
            
            granian = Granian(
                "backend.api:app",
                address=host,
                port=port,
                interface=Interfaces.ASGI,
                workers=int(os.getenv("WORKERS", "1")),
            )
            granian.serve()
            return
            
        except ImportError:
            logging.warning("Granian not available, falling back to Uvicorn")
        except Exception as e:
            logging.warning(f"Granian failed to start: {e}, falling back to Uvicorn")
    
    # Fallback to Uvicorn
    import uvicorn
    logging.info(f"Starting Uvicorn server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

"""
DATect Web Application API
FastAPI backend providing forecasting, visualization, and analysis endpoints
"""

import logging
import math
import os
import re
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel


from forecasting.forecast_engine import ForecastEngine
from forecasting.model_factory import ModelFactory
import config
from backend.visualizations import (
    generate_correlation_heatmap,
    generate_sensitivity_analysis,
    generate_time_series_comparison,
    generate_waterfall_plot,
    generate_spectral_analysis,
    generate_gradient_uncertainty_plot,
)
from backend.cache_manager import cache_manager

# Configure logging
logger = logging.getLogger(__name__)

def clean_float_for_json(value):
    """Handle inf/nan values for JSON serialization"""
    if value is None or (isinstance(value, float) and (math.isinf(value) or math.isnan(value))):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return float(value) if isinstance(value, np.floating) else int(value)
    return value

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isabs(config.FINAL_OUTPUT_PATH):
    config.FINAL_OUTPUT_PATH = os.path.join(project_root, config.FINAL_OUTPUT_PATH)


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
    # Default for local dev; production uses same-origin
    origins = ["http://localhost:3000", "http://localhost:5173", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False if "*" in origins else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Lazy singletons for Cloud Run optimization
forecast_engine = None
model_factory = None

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

def get_model_factory() -> ModelFactory:
    global model_factory
    if model_factory is None:
        model_factory = ModelFactory()
    return model_factory

def get_site_mapping(data):
    """Get site name mapping for flexible API access"""
    return {s.lower().replace(' ', '-'): s for s in data['site'].unique()}

def resolve_site_name(site: str, site_mapping: dict) -> str:
    """Resolve site name from mapping or return original"""
    return site_mapping.get(site.lower(), site)


# Pydantic models
class ForecastRequest(BaseModel):
    date: date
    site: str
    task: str = "regression"  # "regression" or "classification"
    model: str = "xgboost"

class ConfigUpdateRequest(BaseModel):
    forecast_mode: str = "realtime"  # "realtime" or "retrospective" 
    forecast_task: str = "regression"  # "regression" or "classification"
    forecast_model: str = "xgboost"  # "xgboost" or "linear" (linear models)
    selected_sites: List[str] = []  # For retrospective site filtering
    forecast_horizon_weeks: int = 1  # Weeks ahead to forecast from data cutoff

class ForecastResponse(BaseModel):
    success: bool
    forecast_date: date
    site: str
    task: str
    model: str
    prediction: Optional[float] = None
    predicted_category: Optional[int] = None
    training_samples: Optional[int] = None
    feature_importance: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

class SiteInfo(BaseModel):
    sites: List[str]
    date_range: Dict[str, date]
    site_mapping: Dict[str, str]

class ModelInfo(BaseModel):
    available_models: Dict[str, List[str]]
    descriptions: Dict[str, str]


class RetrospectiveRequest(BaseModel):
    selected_sites: List[str] = []  # Empty list means all sites


@app.get("/api")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DATect API - Domoic Acid Forecasting System",
        "version": "1.0.0",
            "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/sites", response_model=SiteInfo)
async def get_sites():
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

@app.get("/api/models", response_model=ModelInfo)
async def get_models():
    """Get available models and their descriptions."""
    try:
        mf = get_model_factory()
        available_models = {
            "regression": mf.get_supported_models('regression')['regression'],
            "classification": mf.get_supported_models('classification')['classification']
        }
        
        # Get descriptions for all models
        descriptions = {}
        all_models = set(available_models["regression"] + available_models["classification"])
        for model in all_models:
            descriptions[model] = mf.get_model_description(model)
        
        return ModelInfo(available_models=available_models, descriptions=descriptions)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model information: {str(e)}")

@app.post("/api/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """Generate a forecast for the specified parameters."""
    try:
        if request.task not in ["regression", "classification"]:
            raise HTTPException(status_code=400, detail="Task must be 'regression' or 'classification'")
        
        data = get_data_copy()
        site_mapping = get_site_mapping(data)
        actual_site = resolve_site_name(request.site, site_mapping)
        
        forecast_date = pd.to_datetime(request.date)
        
        result = get_forecast_engine().generate_single_forecast(
            config.FINAL_OUTPUT_PATH,
            forecast_date,
            actual_site,
            request.task,
            "xgboost"
        )
        
        if result is None:
            return ForecastResponse(
                success=False,
                forecast_date=forecast_date.date(),
                site=request.site,
                task=request.task,
                model=request.model,
                error="Insufficient data for forecast"
            )
        
        # Format response based on task type
        response_data = {
            "success": True,
            "forecast_date": forecast_date.date(),
            "site": request.site,
            "task": request.task,
            "model": request.model,
            "training_samples": result.get('training_samples')
        }
        
        if request.task == "regression":
            response_data["prediction"] = result.get('predicted_da')
        elif request.task == "classification":
            response_data["predicted_category"] = result.get('predicted_category')
        
        if 'feature_importance' in result and result['feature_importance'] is not None:
            importance_df = result['feature_importance']
            if hasattr(importance_df, 'to_dict'):
                response_data["feature_importance"] = importance_df.head(10).to_dict('records')
        
        return ForecastResponse(**response_data)
        
    except Exception as e:
        # Use request date if forecast_date wasn't calculated due to early error
        error_date = request.date
        if 'forecast_date' in locals():
            error_date = forecast_date.date()
        
        return ForecastResponse(
            success=False,
            forecast_date=error_date,
            site=request.site,
            task=request.task,
            model=request.model,
            error=str(e)
        )

@app.get("/api/historical/{site}")
async def get_historical_data(
    site: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = 1000
):
    """Get historical DA measurements for a site."""
    try:
        # Load data
        data = get_data_copy()
        data['date'] = pd.to_datetime(data['date'])
        
        site_mapping = get_site_mapping(data)
        actual_site = resolve_site_name(site, site_mapping)
        
        # Filter by site
        site_data = data[data['site'] == actual_site].copy()
        
        if site_data.empty:
            raise HTTPException(status_code=404, detail=f"Site '{site}' not found")
        
        # Apply date filters
        if start_date:
            site_data = site_data[site_data['date'] >= pd.to_datetime(start_date)]
        if end_date:
            site_data = site_data[site_data['date'] <= pd.to_datetime(end_date)]
        
        # Limit results and sort by date
        site_data = site_data.sort_values('date').tail(limit)
        
        # Build canonical-only payload
        canonical = pd.DataFrame()
        canonical['date'] = site_data['date'].dt.strftime('%Y-%m-%d')
        canonical['actual_da'] = site_data['da']
        if 'da-category' in site_data.columns:
            canonical['actual_category'] = site_data['da-category']

        return {
            "site": site,
            "count": len(canonical),
            "data": canonical.to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load historical data: {str(e)}")

@app.get("/api/config")
async def get_config():
    """Get current system configuration."""
    return {
        "forecast_mode": getattr(config, 'FORECAST_MODE', 'realtime'),
        "forecast_task": getattr(config, 'FORECAST_TASK', 'regression'),
        "forecast_model": getattr(config, 'FORECAST_MODEL', 'xgboost'),
        "forecast_horizon_weeks": getattr(config, 'FORECAST_HORIZON_WEEKS', 1),
        "forecast_horizon_days": getattr(config, 'FORECAST_HORIZON_DAYS', 7)
    }

@app.post("/api/config")
async def update_config(config_request: ConfigUpdateRequest):
    """Update system configuration and write to config.py file."""
    try:
        # Update in-memory config values
        config.FORECAST_MODE = config_request.forecast_mode
        config.FORECAST_TASK = config_request.forecast_task  
        config.FORECAST_MODEL = config_request.forecast_model
        config.FORECAST_HORIZON_WEEKS = config_request.forecast_horizon_weeks
        config.FORECAST_HORIZON_DAYS = config_request.forecast_horizon_weeks * 7
        
        # Write changes to config.py file
        config_file_path = os.path.join(project_root, 'config.py')
        
        # Read current config.py
        with open(config_file_path, 'r') as f:
            config_content = f.read()
        
        # Update the specific lines
        config_content = re.sub(
            r'FORECAST_MODE = ".*?"',
            f'FORECAST_MODE = "{config_request.forecast_mode}"',
            config_content
        )
        config_content = re.sub(
            r'FORECAST_TASK = ".*?"',
            f'FORECAST_TASK = "{config_request.forecast_task}"',
            config_content
        )
        config_content = re.sub(
            r'FORECAST_MODEL = ".*?"',
            f'FORECAST_MODEL = "{config_request.forecast_model}"',
            config_content
        )
        config_content = re.sub(
            r'FORECAST_HORIZON_WEEKS = \d+',
            f'FORECAST_HORIZON_WEEKS = {config_request.forecast_horizon_weeks}',
            config_content
        )
        config_content = re.sub(
            r'FORECAST_HORIZON_DAYS = .*',
            f'FORECAST_HORIZON_DAYS = FORECAST_HORIZON_WEEKS * 7  # Derived days value for internal calculations',
            config_content
        )
        
        # Write back to file
        with open(config_file_path, 'w') as f:
            f.write(config_content)
        
        return {
            "success": True,
            "message": "Configuration updated successfully in config.py",
            "config": {
                "forecast_mode": config.FORECAST_MODE,
                "forecast_task": config.FORECAST_TASK,
                "forecast_model": config.FORECAST_MODEL
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

@app.get("/api/historical/all")
async def get_all_sites_historical(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: Optional[int] = 1000
):
    """Get historical data for all sites."""
    try:
        data = get_data_copy()
        
        # Apply date filters
        if start_date:
            data = data[pd.to_datetime(data['date']) >= pd.to_datetime(start_date)]
        if end_date:
            data = data[pd.to_datetime(data['date']) <= pd.to_datetime(end_date)]
        
        # Sort by date and site
        data = data.sort_values(['site', 'date'])
        
        # Limit results
        if limit:
            data = data.head(limit)
        
        # Convert to dict for JSON response with proper float cleaning (canonical-only)
        result = []
        for _, row in data.iterrows():
            da_val = row['da'] if pd.notna(row['da']) else None
            da_category = row['da-category'] if 'da-category' in row and pd.notna(row['da-category']) else None
            
            rec = {
                'date': row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else None,
                'site': row['site'],
                'actual_da': clean_float_for_json(da_val),
            }
            if da_category is not None:
                rec['actual_category'] = clean_float_for_json(da_category)
            result.append(rec)
        
        return JSONResponse(content={
            "success": True,
            "data": result,
            "count": len(result)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve historical data: {str(e)}")

# Visualization endpoints
# NOTE: More specific routes must come before generic parameter routes
@app.get("/api/visualizations/correlation/all")
async def get_correlation_heatmap_all():
    """Generate correlation heatmap for all sites combined."""
    try:
        data = get_data_copy()
        plot_data = generate_correlation_heatmap(data, site=None)
        return {"success": True, "plot": plot_data, "cached": False, "source": "computed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate correlation heatmap: {str(e)}")

@app.get("/api/visualizations/correlation/{site}")
async def get_correlation_heatmap_single(site: str):
    """Generate correlation heatmap for a single site."""
    try:
        data = get_data_copy()
        
        site_mapping = get_site_mapping(data)
        actual_site = resolve_site_name(site, site_mapping)

        cached_plot = cache_manager.get_correlation_heatmap(actual_site)
        if cached_plot is not None:
            return {"success": True, "plot": cached_plot, "cached": True, "source": "precomputed"}

        plot_data = generate_correlation_heatmap(data, actual_site)
        return {"success": True, "plot": plot_data, "cached": False, "source": "computed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate correlation heatmap: {str(e)}")

@app.get("/api/visualizations/sensitivity/all")
async def get_sensitivity_analysis_all():
    """Generate sensitivity analysis plots for all sites combined."""
    try:
        data = get_data_copy()
        plots = generate_sensitivity_analysis(data, site=None)
        return {"success": True, "plots": plots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate sensitivity analysis: {str(e)}")

@app.get("/api/visualizations/sensitivity/{site}")
async def get_sensitivity_analysis_single(site: str):
    """Generate sensitivity analysis plots for a single site."""
    try:
        data = get_data_copy()
        
        site_mapping = get_site_mapping(data)
        actual_site = resolve_site_name(site, site_mapping)
        
        plots = generate_sensitivity_analysis(data, actual_site)
        return {"success": True, "plots": plots}
    except Exception as e:
        logging.error(f"Error in sensitivity analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate sensitivity analysis: {str(e)}")

@app.get("/api/visualizations/comparison/all")
async def get_time_series_comparison_all():
    """Generate time series comparison for all sites."""
    try:
        data = get_data_copy()
        plot_data = generate_time_series_comparison(data, site=None)
        return {"success": True, "plot": plot_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate time series comparison: {str(e)}")

@app.get("/api/visualizations/comparison/{site}")
async def get_time_series_comparison_single(site: str):
    """Generate time series comparison for a single site."""
    try:
        data = get_data_copy()
        
        site_mapping = get_site_mapping(data)
        actual_site = resolve_site_name(site, site_mapping)
        
        plot_data = generate_time_series_comparison(data, actual_site)
        return {"success": True, "plot": plot_data}  # plot_data is already in correct format
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate time series comparison: {str(e)}")

@app.get("/api/visualizations/waterfall")
async def get_waterfall_plot():
    """Generate waterfall plot for all sites."""
    try:
        data = get_data_copy()
        plot_data = generate_waterfall_plot(data)
        return {"success": True, "plot": plot_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate waterfall plot: {str(e)}")

@app.get("/api/visualizations/spectral/all")
async def get_spectral_analysis_all():
    """Generate spectral analysis for all sites combined (uses pre-computed cache)."""
    try:
        # First try pre-computed cache
        plots = cache_manager.get_spectral_analysis(site=None)
        
        if plots is not None:
            return {"success": True, "plots": plots, "cached": True, "source": "precomputed"}

        # Compute on server (expensive - only for local development)
        logging.warning("Computing spectral analysis on server - this is very expensive!")
        data = get_data_copy()
        plots = generate_spectral_analysis(data, site=None)
        return {"success": True, "plots": plots, "cached": False, "source": "computed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate spectral analysis: {str(e)}")

@app.get("/api/visualizations/spectral/{site}")
async def get_spectral_analysis_single(site: str):
    """Generate spectral analysis for a single site (uses pre-computed cache)."""
    try:
        data = get_data_copy()
        site_mapping = get_site_mapping(data)
        actual_site = resolve_site_name(site, site_mapping)

        # First try pre-computed cache
        plots = cache_manager.get_spectral_analysis(site=actual_site)
        
        if plots is not None:
            return {"success": True, "plots": plots, "cached": True, "source": "precomputed"}

        # Compute on server (expensive - only for local development)
        logging.warning(f"Computing spectral analysis for {actual_site} on server - this is very expensive!")
        plots = generate_spectral_analysis(data, actual_site)
        return {"success": True, "plots": plots, "cached": False, "source": "computed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate spectral analysis: {str(e)}")

def get_latest_da_from_raw_files():
    """Get the latest DA measurements from raw CSV files."""
    import os
    from datetime import datetime
    
    raw_da_dir = "./data/raw/da-input"
    latest_da_data = {}
    
    # Site mapping from file names to display names
    site_file_mapping = {
        'cannon-beach': 'Cannon Beach',
        'clatsop-beach': 'Clatsop Beach', 
        'coos-bay': 'Coos Bay',
        'copalis': 'Copalis',
        'gold-beach': 'Gold Beach',
        'kalaloch': 'Kalaloch',
        'long-beach': 'Long Beach',
        'newport': 'Newport',
        'quinault': 'Quinault', 
        'twin-harbors': 'Twin Harbors'
    }
    
    for file_key, site_name in site_file_mapping.items():
        file_path = os.path.join(raw_da_dir, f"{file_key}-da.csv")
        
        if not os.path.exists(file_path):
            continue
            
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            if df.empty:
                continue
                
            # Get the last row (most recent entry)
            last_row = df.iloc[-1]
            
            # Parse date and DA value based on format
            if 'CollectDate' in df.columns and 'Domoic Result' in df.columns:
                # Format B: CollectDate,Domoic Result
                date_str = str(last_row['CollectDate'])
                da_value = last_row['Domoic Result']
                
                # Parse date (format like "11/30/2023")
                try:
                    date_obj = pd.to_datetime(date_str)
                except:
                    date_obj = datetime.now()
                    
            elif 'Harvest Month' in df.columns and 'Harvest Date' in df.columns and 'Harvest Year' in df.columns:
                # Format A: Harvest Month,Harvest Date,Harvest Year,Domoic Acid
                month = str(last_row['Harvest Month'])
                day = str(last_row['Harvest Date'])
                year = str(last_row['Harvest Year'])
                da_value = last_row['Domoic Acid']
                
                # Parse date
                try:
                    date_str = f"{month} {day}, {year}"
                    date_obj = pd.to_datetime(date_str)
                except:
                    date_obj = datetime.now()
            else:
                continue
            
            # Handle DA value (might be "<1" or other string formats)
            try:
                if isinstance(da_value, str):
                    if '<' in da_value:
                        # Handle "<1" as 0.5
                        da_numeric = float(da_value.replace('<', '')) / 2
                    else:
                        da_numeric = float(da_value)
                else:
                    da_numeric = float(da_value)
            except:
                da_numeric = 0.0
            
            latest_da_data[site_name] = {
                'da': da_numeric,
                'date': date_obj
            }
            
        except Exception as e:
            logging.warning(f"Failed to parse {file_path}: {e}")
            continue
    
    return latest_da_data

@app.get("/api/visualizations/map")
async def get_site_map():
    """Generate map visualization of all 10 monitoring sites with risk level colors using latest raw data."""
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
                
                # Determine risk level
                if recent_da < 5:
                    risk_level = "Low"
                    color = "green"
                elif recent_da < 20:
                    risk_level = "Moderate"
                    color = "yellow"
                elif recent_da < 40:
                    risk_level = "High"
                    color = "orange"
                else:
                    risk_level = "Extreme"
                    color = "red"
                
                hover_text = f'<b>{site_name}</b><br>' + \
                            f'Latitude: {lat:.4f}<br>' + \
                            f'Longitude: {lon:.4f}<br>' + \
                            f'Recent DA Level: {recent_da:.2f} μg/g<br>' + \
                            f'Risk Level: {risk_level}<br>' + \
                            f'Date: {recent_date.strftime("%Y-%m-%d") if hasattr(recent_date, "strftime") else str(recent_date)}<extra></extra>'
            else:
                # No data available - use default color
                color = "gray"
                risk_level = "No Data"
                hover_text = f'<b>{site_name}</b><br>' + \
                            f'Latitude: {lat:.4f}<br>' + \
                            f'Longitude: {lon:.4f}<br>' + \
                            f'Risk Level: {risk_level}<extra></extra>'
            
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
            title='DATect Monitoring Sites - Pacific Coast (Risk Levels)',
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



@app.post("/api/forecast/enhanced")
async def generate_enhanced_forecast(request: ForecastRequest):
    """Generate enhanced forecast with both regression and classification for frontend."""
    try:
        data = get_data_copy()
        site_mapping = get_site_mapping(data)
        actual_site = resolve_site_name(request.site, site_mapping)
        
        forecast_date = pd.to_datetime(request.date)
        
        # Generate regression and classification forecasts
        regression_result = get_forecast_engine().generate_single_forecast(
            config.FINAL_OUTPUT_PATH, forecast_date, actual_site, "regression", "xgboost"
        )
        
        classification_result = get_forecast_engine().generate_single_forecast(
            config.FINAL_OUTPUT_PATH, forecast_date, actual_site, "classification", "xgboost"
        )
        
        # Clean numpy values for JSON serialization
        def clean_numpy_values(obj):
            if isinstance(obj, dict):
                return {k: clean_numpy_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_numpy_values(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, pd.DataFrame):
                return obj.head(10).to_dict('records') if not obj.empty else []
            elif obj is None:
                return None
            else:
                try:
                    if pd.isna(obj):
                        return None
                except (TypeError, ValueError):
                    pass
                return obj
        
        # Create response structure expected by frontend
        response_data = {
            "success": True,
            "forecast_date": forecast_date.strftime('%Y-%m-%d'),
            "site": actual_site,
            "regression": clean_numpy_values(regression_result),
            "classification": clean_numpy_values(classification_result),
            "graphs": {}
        }
        
        # Add level_range graph for regression
        if regression_result and 'predicted_da' in regression_result:
            predicted_da = float(regression_result['predicted_da'])
            
            # Use bootstrap confidence intervals if available, otherwise fall back to simple multipliers
            bootstrap_quantiles = regression_result['bootstrap_quantiles']
            quantiles = {
                "q05": bootstrap_quantiles['q05'],
                "q50": bootstrap_quantiles['q50'],
                "q95": bootstrap_quantiles['q95'],
            }
            # Provide a robust gradient plot (handles degenerate quantile ranges)
            gradient_plot_json = generate_gradient_uncertainty_plot(quantiles, predicted_da)

            response_data["graphs"]["level_range"] = {
                "gradient_quantiles": quantiles,
                "xgboost_prediction": predicted_da,
                "type": "gradient_uncertainty",
                "gradient_plot": gradient_plot_json,
            }
        
        # Add category_range graph for classification
        if classification_result and 'predicted_category' in classification_result:
            class_probs = classification_result.get('class_probabilities', [0.25, 0.25, 0.25, 0.25])
            if isinstance(class_probs, np.ndarray):
                class_probs = class_probs.tolist()
            
            response_data["graphs"]["category_range"] = {
                "predicted_category": int(classification_result['predicted_category']),
                "class_probabilities": class_probs,
                "category_labels": ['Low (≤5)', 'Moderate (5-20]', 'High (20-40]', 'Extreme (>40)'],
                "type": "category_range"
            }
        
        
        return response_data
        
    except Exception as e:
        return {
            "success": False,
            "forecast_date": request.date,
            "site": request.site,
            "error": str(e)
        }



@app.post("/api/retrospective")
async def run_retrospective_analysis(request: RetrospectiveRequest = RetrospectiveRequest()):
    """Run complete retrospective analysis based on current config (uses pre-computed cache for production)."""
    try:
        # Map model names for API compatibility
        if config.FORECAST_MODEL == "linear":
            actual_model = "linear" if config.FORECAST_TASK == "regression" else "logistic"
        else:
            actual_model = config.FORECAST_MODEL
        
        # First try to get from pre-computed cache (for production)
        base_results = cache_manager.get_retrospective_forecast(config.FORECAST_TASK, actual_model)
        
        if base_results is None:
            # Compute on server (expensive - only for local development)
            logging.warning("Computing retrospective analysis on server - this is expensive!")
            engine = get_forecast_engine()
            engine.data_file = config.FINAL_OUTPUT_PATH
            
            results_df = engine.run_retrospective_evaluation(
                task=config.FORECAST_TASK,
                model_type=actual_model,
                n_anchors=getattr(config, 'N_RANDOM_ANCHORS', 500)
            )

            if results_df is None or results_df.empty:
                return {"success": False, "error": "No results generated from retrospective analysis"}

            # Convert results to JSON format with proper float cleaning (canonical keys)
            base_results = []
            for _, row in results_df.iterrows():
                record = {
                    "date": row['date'].strftime('%Y-%m-%d') if pd.notnull(row['date']) else None,
                    "site": row['site'],
                    "actual_da": clean_float_for_json(row['actual_da']) if 'actual_da' in row and pd.notnull(row['actual_da']) else None,
                    "predicted_da": clean_float_for_json(row['predicted_da']) if 'predicted_da' in row and pd.notnull(row['predicted_da']) else None,
                    "actual_category": clean_float_for_json(row['actual_category']) if 'actual_category' in row and pd.notnull(row['actual_category']) else None,
                    "predicted_category": clean_float_for_json(row['predicted_category']) if 'predicted_category' in row and pd.notnull(row['predicted_category']) else None
                }
                if 'anchor_date' in results_df.columns and pd.notnull(row.get('anchor_date', None)):
                    record['anchor_date'] = row['anchor_date'].strftime('%Y-%m-%d')
                base_results.append(record)

            # Results computed on-demand for local development
        else:
            logging.info(f"Serving pre-computed retrospective analysis: {config.FORECAST_TASK}+{actual_model}")

        # Cached data now uses the same standardized format as fresh computation (actual_da, predicted_da, actual_category, predicted_category)

        # Filter by sites if specified
        filtered = [r for r in base_results if r['site'] in request.selected_sites] if request.selected_sites else base_results
        
        summary = _compute_summary(filtered)
        return {
            "success": True,
            "config": {
                "forecast_mode": config.FORECAST_MODE,
                "forecast_task": config.FORECAST_TASK,
                "forecast_model": config.FORECAST_MODEL
            },
            "summary": summary,
            "results": filtered
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def _compute_summary(results_json: list) -> dict:
    """Compute summary metrics for retrospective results using canonical keys only."""
    summary = {"total_forecasts": len(results_json)}

    # Extract valid pairs strictly from canonical keys
    valid_regression = []
    valid_classification = []
    for r in results_json:
        a = r.get('actual_da')
        p = r.get('predicted_da')
        if a is not None and p is not None:
            valid_regression.append((a, p))

        ac = r.get('actual_category')
        pc = r.get('predicted_category')
        if ac is not None and pc is not None:
            valid_classification.append((ac, pc))

    summary["regression_forecasts"] = len(valid_regression)
    summary["classification_forecasts"] = len(valid_classification)

    # Regression metrics
    if valid_regression:
        from sklearn.metrics import r2_score, mean_absolute_error, f1_score
        actual_vals = [r[0] for r in valid_regression]
        pred_vals = [r[1] for r in valid_regression]
        try:
            summary["r2_score"] = float(r2_score(actual_vals, pred_vals))
            summary["mae"] = float(mean_absolute_error(actual_vals, pred_vals))

            # F1 score for spike detection (20 μg/g threshold)
            spike_threshold = 20.0
            actual_binary = [1 if val > spike_threshold else 0 for val in actual_vals]
            pred_binary = [1 if val > spike_threshold else 0 for val in pred_vals]
            summary["f1_score"] = float(
                f1_score(actual_binary, pred_binary, zero_division=0)
            )
        except Exception:
            pass

    # Classification metrics
    if valid_classification:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
        actual_cats = [r[0] for r in valid_classification]
        pred_cats = [r[1] for r in valid_classification]
        try:
            summary["accuracy"] = float(accuracy_score(actual_cats, pred_cats))
            summary["balanced_accuracy"] = float(balanced_accuracy_score(actual_cats, pred_cats))
            
            # Per-class metrics
            classes = [0, 1, 2, 3]
            class_names = ['Low', 'Moderate', 'High', 'Extreme']
            per_class_metrics = {}
            
            precision, recall, f1, support = precision_recall_fscore_support(
                actual_cats, pred_cats, labels=classes, zero_division=0
            )
            
            for i, class_name in enumerate(class_names):
                per_class_metrics[class_name] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                }
            
            summary["per_class_metrics"] = per_class_metrics
            
        except Exception as e:
            logging.error(f"Error calculating classification metrics: {e}")

    return summary

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

# DATect Dependencies

## Python: Used vs Optional

### Core (required)
| Package | Used in |
|---------|---------|
| pandas, numpy | Everywhere |
| scikit-learn | data_processor, model_factory, visualizations |
| joblib | forecast_engine (parallel) |
| xgboost | model_factory (GPU auto-detected via tree_method='gpu_hist') |
| fastapi, uvicorn, pydantic | backend/api |
| plotly | backend/visualizations, api map |
| xarray, netcdf4 | dataset-creation (NetCDF) |
| requests | run_datect, dataset-creation |
| scipy | visualizations (signal, spectral) |
| tqdm | forecast_engine, dataset-creation |
| pyarrow | Parquet I/O |
| SALib | visualizations (Sobol sensitivity) |
| python-dateutil | pandas date parsing |

### Optional (graceful fallback)
| Package | Used in | If missing |
|---------|---------|------------|
| polars | data_processor | Falls back to pandas |
| duckdb | data_processor | Falls back to pyarrow |
| numba | data_processor, dataset-creation | Falls back to pure Python |
| granian | run_datect, api | Falls back to uvicorn |
| redis | backend/redis_cache, cache_manager | File cache only |

### Not used in current code
| Package | Note |
|---------|------|
| lightgbm | Model factory only has XGBoost + linear |
| catboost | Not imported |
| optuna | hyperparam_search uses manual random search |
| pandera | No schema validation in code |
| matplotlib | Only plotly is used |
| httpx | Only `requests` is used |

These are in optional groups in pyproject.toml so you can install them for experiments without bloating the default install.

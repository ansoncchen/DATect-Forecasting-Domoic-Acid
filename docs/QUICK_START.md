# DATect - Quick Start Guide

This guide covers setting up DATect for local development and Google Cloud deployment.

## Prerequisites

| Software | Version | Installation |
|----------|---------|--------------|
| Python | 3.8+ | [python.org](https://www.python.org/downloads/) |
| Node.js | 16+ (18+ recommended) | [nodejs.org](https://nodejs.org/) |
| Git | Any | [git-scm.com](https://git-scm.com/) |

### macOS

```bash
brew install python node git
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip nodejs npm git

# For newer Node.js:
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### Windows

Download and install from the links above. Check "Add Python to PATH" during installation.

## Local Setup

```bash
# Clone repository
git clone https://github.com/your-username/DATect-Forecasting-Domoic-Acid.git
cd DATect-Forecasting-Domoic-Acid

# Run (auto-installs dependencies)
python run_datect.py
```

The launcher will:
1. Check prerequisites
2. Install Python and Node.js dependencies
3. Verify the dataset exists
4. Start backend API (port 8000) and frontend (port 3000)
5. Open browser to http://localhost:3000

### First Run with Missing Dataset

If the dataset doesn't exist, you'll need to generate it first:

```bash
python dataset-creation.py  # Takes 30-60 minutes
python run_datect.py        # Then start the system
```

## Google Cloud Deployment

### Setup Google Cloud

```bash
# Install Google Cloud CLI (varies by OS)
# macOS: brew install google-cloud-sdk
# Linux: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Create project (or use existing)
gcloud projects create your-datect-project --name="DATect"
gcloud config set project your-datect-project

# Enable services
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

### Deploy

```bash
# Ensure dataset exists locally
python dataset-creation.py

# Deploy to Cloud Run
./deploy_gcloud.sh
```

The deployment will build a container and deploy to Cloud Run.

## Using the System

### Dashboard

1. **Select date**: Choose forecast target date (2008-2024)
2. **Select site**: Pick from 10 Pacific Coast locations
3. **Select model**: XGBoost (recommended) or Linear
4. **Click Forecast**: View prediction with confidence intervals

### Risk Categories

| Category | DA Level (μg/g) | Meaning |
|----------|-----------------|---------|
| Low | 0-5 | Safe for consumption |
| Moderate | 5-20 | Caution advised |
| High | 20-40 | Avoid consumption |
| Extreme | >40 | Health hazard |

### Historical Analysis

Access visualization tools:
- **Correlation heatmaps**: Variable relationships
- **Sensitivity analysis**: Feature importance (Sobol indices)
- **Time series**: DA vs Pseudo-nitzschia over time
- **Spectral analysis**: Frequency domain patterns

## Troubleshooting

### "Command not found: python"

```bash
# Try python3
python3 run_datect.py

# Check installation
which python3
python3 --version
```

### Port Already in Use

`run_datect.py` automatically kills existing processes on ports 8000/3000. If issues persist:

```bash
# Manual kill
lsof -ti:8000,3000 | xargs kill -9
```

### Node.js Version Too Old

```bash
# Check version
node --version  # Should be 16+

# Update (macOS)
brew upgrade node

# Update (Linux)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### Google Cloud Deployment Failed

```bash
# Check authentication
gcloud auth list
gcloud auth application-default login

# Verify project
gcloud config list

# Enable APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

## Verification

After setup, verify the system works:

```bash
# Pre-compute cache (includes validation)
python precompute_cache.py

# Check API health
curl http://localhost:8000/health
# Should return: {"status": "healthy"}

# Test forecast via web interface
# Open http://localhost:3000 and generate a forecast
```

## Dependencies

### Python (auto-installed)

- **Scientific**: pandas, numpy, scipy, scikit-learn, xgboost
- **High-performance**: polars, duckdb, pyarrow
- **Web**: fastapi, uvicorn, pydantic
- **Data**: xarray, netcdf4, requests
- **Visualization**: plotly, matplotlib

### Node.js (auto-installed)

- **Framework**: React 18, Vite
- **Visualization**: Plotly.js
- **Styling**: TailwindCSS

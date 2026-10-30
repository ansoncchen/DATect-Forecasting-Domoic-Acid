# Deployment Guide

DATect deploys as one container on Google Cloud Run: the FastAPI backend serves the API and the built React app from the same origin, with the pre-computed cache baked into the image.

```
Hyak                         Your machine                     Google Cloud
precompute_cache.py   ──→    copy cache/ into the repo  ──→   deploy/deploy_gcloud.sh
                                                              (Cloud Build → Cloud Run)
```

## 1. Build the cache on Hyak

Only needed when model code, configuration or data changed. From the repo root on Hyak (see [`HYAK_SETUP.md`](HYAK_SETUP.md)):

```bash
$DATECT_PY precompute_cache.py
```

It writes:

| Path | Contents |
|---|---|
| `cache/retrospective/` | 8 task/model retrospectives, each as `.parquet` + `.json` (16 files): regression × {ensemble, xgboost, rf, naive, linear}, classification × {ensemble, naive, logistic} |
| `cache/spectral/` | spectral analysis per site plus `all_sites.json` |
| `cache/visualizations/` | per-site correlation heatmaps (`<site>_correlation.json`) |
| `cache/manifest.json` | file list |

`cache/` is git-ignored.

## 2. Copy the cache to your machine

```bash
cd /path/to/DATect-Forecasting-Domoic-Acid
rsync -avz klone-login:/gscratch/stf/YOUR_UWNETID/DATect-Forecasting-Domoic-Acid/cache/ ./cache/
```

Check it locally before deploying:

```bash
ENABLE_PRECOMPUTED_CACHE=true python3 run_datect.py
```

## 3. Deploy

```bash
gcloud auth login                      # once
export PROJECT_ID=your-gcp-project-id  # required
export REGION=us-west1                 # optional, default us-west1
./deploy/deploy_gcloud.sh              # it cd's to the repo root itself
```

What the script does:

1. If `./cache` is missing or empty it runs `precompute_cache.py` **locally**, which is far too heavy for a laptop; copy the Hyak cache first. If the cache exists it asks "Re-generate cache? (y/N)": answer **N**.
2. Sets the gcloud project and region and enables the Cloud Build, Cloud Run and Container Registry APIs.
3. Runs `gcloud builds submit --config=deploy/cloudbuild.yaml`. The upload is limited by `.gcloudignore` to the build inputs.
4. Cloud Build (`deploy/cloudbuild.yaml`) builds `Dockerfile.production`, pushes `gcr.io/$PROJECT_ID/datect:latest`, and deploys the Cloud Run service `datect-api`.
5. Reads the service URL and checks `GET /health`.

Cloud Run settings (from `deploy/cloudbuild.yaml`): `--allow-unauthenticated`, 1 GiB memory, 1 CPU, min 0 / max 3 instances, concurrency 20, 300 s request timeout, env `CACHE_DIR=/app/cache, NODE_ENV=production, PYTHONPATH=/app`. With min instances 0 the service scales to zero, so the first request after idle is a cold start.

> **Container Registry is deprecated.** The build pushes to `gcr.io` and the script enables `containerregistry.googleapis.com`. Google is retiring Container Registry in favor of Artifact Registry (`REGION-docker.pkg.dev/PROJECT/REPOSITORY/IMAGE`); migrating `deploy/cloudbuild.yaml` and `deploy/deploy_gcloud.sh` is an open follow-up.

## The production image (`Dockerfile.production`)

Three stages:

1. **Python deps** (`python:3.11-slim`): `uv pip install -r requirements.txt`.
2. **Frontend** (`node:22-slim`): `npm ci`, `npm run build` → `frontend/dist`.
3. **Runtime** (`python:3.11-slim`): copies `backend/`, `forecasting/`, `data/`, `config/` (needed: `config.py` loads `config/tuned_hyperparameters.json` at import), `config.py`, `results/chronological` and `results/multi_seed` (read by `GET /api/metrics/summary`), `cache/`, and the built frontend. The build fails early if `cache/` or `frontend/dist` is missing or `import fastapi, granian, config` fails. Runs as a non-root user on port 8080 with a `/health` healthcheck; starts Granian (`DATECT_USE_GRANIAN=1`, the default) or Uvicorn.

Build inputs are whitelisted: `.dockerignore` (Docker build context) and `.gcloudignore` (Cloud Build upload) admit only what the Dockerfile copies. A new directory the image needs must be added to the Dockerfile **and** both ignore files.

Run the image locally:

```bash
docker build -f Dockerfile.production -t datect .
docker run -p 8080:8080 datect          # http://localhost:8080
```

**CI** (`.github/workflows/docker.yml`): on pushes and pull requests to `main` (except docs/papers-only changes) it builds the image with a stub `cache/` and fails unless `/health` answers within about a minute. It catches missing `COPY` lines and import-time crashes that unit tests cannot see.

## Environment variables

Read by the server code (`backend/`), `run_datect.py` and the image:

| Variable | Read by | Effect |
|---|---|---|
| `ENABLE_PRECOMPUTED_CACHE` | `backend/cache_manager.py` | `true` serves the file cache. Needed for local runs; in the image the cache is already enabled by `NODE_ENV`/`CACHE_DIR`. |
| `CACHE_DIR` | `backend/cache_manager.py`, `run_datect.py` | Cache location (default `./cache`; `run_datect.py` defaults it to `<repo>/cache`). The value `/app/cache` also enables the cache. |
| `NODE_ENV` | `backend/cache_manager.py` | `production` enables the cache. Set in the image and on Cloud Run. |
| `REDIS_URL` | `backend/cache_manager.py`, `backend/redis_cache.py` | Optional Redis layer in front of the file cache. |
| `ALLOWED_ORIGINS` | `backend/api.py` | Comma-separated CORS origins. Unset → `http://localhost:3000`, `http://localhost:5173`. Not needed for the single-origin Cloud Run deploy; set it if another origin calls the API. |
| `PORT` | Docker `CMD`, `backend/api.py` | Listen port: 8080 in the image; `python -m backend.api` defaults to 8000. |
| `DATECT_USE_GRANIAN` | Docker `CMD`, `backend/api.py` | `1` (default) Granian, otherwise Uvicorn. |
| `WORKERS` | `backend/api.py` | Granian worker count, only when started with `python -m backend.api` (default 1). The Docker `CMD` does not read it. |
| `VITE_API_URL` | `frontend/src/services/api.js` (build time) | API base URL baked into the frontend. Unset → same origin (`/api`); the Vite dev server proxies `/api` to `localhost:8000`. |

The engine's `DATECT_*` overrides (see `config.py`) apply to the API process too; the deployed service uses none.

There is no runtime configuration endpoint: `POST /api/config` was removed, and task and model now travel with each request (`/api/retrospective` takes them in the body). `GET /api/config` remains read-only.

## Verify and operate

```bash
curl https://SERVICE_URL/health
curl https://SERVICE_URL/api/metrics/summary
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=datect-api" \
  --limit 50 --project "$PROJECT_ID"
gcloud run services delete datect-api --platform managed --region "$REGION"   # tear down
```

Set a billing budget alert in the Cloud console to catch unexpected charges.

| Problem | Fix |
|---|---|
| `PROJECT_ID environment variable is required` | `export PROJECT_ID=...` |
| Script starts `precompute_cache.py` locally | `cache/` was missing or empty; stop it and copy the Hyak cache. |
| Docker build fails at `ls -la ./cache/` | `cache/` absent from the build context. |
| Container exits at startup | Check the logs; startup errors reach them (no stderr redirection). A missing `COPY` (e.g. `config/`) shows up as an import error. |
| `/api/metrics/summary` returns 404 | `results/chronological` missing from the image (without `results/multi_seed` the multi-seed fields are empty). |
| Retrospective or spectral views are slow | The cache is not enabled or not populated; the API is computing on demand. |

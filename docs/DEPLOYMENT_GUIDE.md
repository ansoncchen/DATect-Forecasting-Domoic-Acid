# Deployment Guide

DATect deploys as one container on Google Cloud Run: the FastAPI backend serves the API and the built React app from the same origin, with the precomputed store baked into the image.

```
Hyak (Spark)                          Your machine                      Google Cloud
datect_spark/store.sbatch      ──→    copy the store into ./store  ──→  deploy/deploy_gcloud.sh
  → ~/datect-store/<commit>/                                           (Cloud Build → Cloud Run)
```

## 1. Build the store on Hyak

The store is a directory of four Delta tables that the API reads with `deltalake` (no JVM) and serves from memory:

| Table | Contents | Built by |
|---|---|---|
| `forecasts` | `generate_single_forecast` output for every (site, week) in `final_output.parquet`, model `ensemble`, regression only (10,950 entries; the measured 2026-11 run stored both tasks, 21,900) | `datect_spark/forecast_serving.py precompute` |
| `retrospective` | `/api/retrospective` records for 8 task/model slices: regression × {ensemble, xgboost, rf, naive, linear}, classification × {ensemble, naive, logistic} | `datect_spark/precompute_store.py retrospective` |
| `spectral` | spectral analysis per site and for all sites | `datect_spark/precompute_store.py visuals` |
| `correlation` | correlation heatmap per site | `datect_spark/precompute_store.py visuals` |

Every row carries a **config hash**: a hash of `forecasting/*.py`, `config.py`, `backend/visualizations.py`, `backend/records.py` (how rows are written), the tuned JSON, `final_output.parquet`, the raw DA CSVs, the `DATECT_*` variables the engine reads, the model-library versions, the CPU architecture and the Python version. The API serves only rows whose hash equals its own; anything else is computed on demand, and `/health` reports stale rows. **Rebuild the store whenever code, config or data change** — at the commit you deploy. The image pins the model libraries to the versions the store is built with (`constraints.txt`, matching the Hyak Spark env), and uses the same Python (3.10), so a store built on Hyak at the deployed commit matches the image's hash (CI prints both hashes; the image build refuses a stale or incomplete store).

From any Hyak checkout at the commit to deploy (two jobs on two nodes, about 2.5 hours; each saves its tables to `~/datect-store/<commit>/`):

```bash
L="-o $HOME/datect-spark-logs/%x-%j.out -e $HOME/datect-spark-logs/%x-%j.err"
sbatch $L --export=ALL,STEPS="forecasts parity save" datect_spark/store.sbatch
sbatch $L --export=ALL,STEPS="retrospective visuals save" datect_spark/store.sbatch
```

`store/` is git-ignored. Measurements and parity checks: [`datect_spark/RESULTS.md`](../datect_spark/RESULTS.md).

## 2. Copy the store to your machine

```bash
cd /path/to/DATect-Forecasting-Domoic-Acid
rsync -a klone-login:datect-store/$(git rev-parse HEAD)/ ./store/
```

Check it against the image before deploying. A store's rows are served only where the config hash matches: the same code, data, model-library versions (`constraints.txt`), Python version (3.10) and CPU architecture (x86_64) as the Hyak env that built it. On a laptop (arm64, other versions) every row shows as stale on `/health` and requests are computed; check a store against the image instead (`docker build --platform linux/amd64 ...`, whose build fails unless the store is complete and current). The image build runs the same check (`python -m backend.precomputed check ./store`):

```bash
docker build --platform linux/amd64 -f Dockerfile.production -t datect .   # fails on a stale or incomplete store
docker run --platform linux/amd64 -p 8080:8080 datect
curl -s localhost:8080/health | python3 -m json.tool                       # tables, entries, stale rows
```

## 3. Deploy

```bash
gcloud auth login                      # once
export PROJECT_ID=your-gcp-project-id  # required
export REGION=us-west1                 # optional, default us-west1
./deploy/deploy_gcloud.sh              # it cd's to the repo root itself
```

What the script does:

1. Stops if `./store` is missing or empty (build it on Hyak and copy it first). The image build itself fails unless the store is complete and current for the image (`REQUIRE_CURRENT_STORE`, on by default).
2. Sets the gcloud project and region and enables the Cloud Build, Cloud Run and Container Registry APIs.
3. Runs `gcloud builds submit --config=deploy/cloudbuild.yaml`. The upload is limited by `.gcloudignore` to the build inputs.
4. Cloud Build (`deploy/cloudbuild.yaml`) builds `Dockerfile.production`, pushes `gcr.io/$PROJECT_ID/datect:latest`, and deploys the Cloud Run service `datect-api`.
5. Reads the service URL and checks `GET /health`.

Cloud Run settings (from `deploy/cloudbuild.yaml`): `--allow-unauthenticated`, 1 GiB memory, 1 CPU, min 0 / max 3 instances, concurrency 20, 300 s request timeout, env `DATECT_PRECOMPUTED_DIR=/app/store, PYTHONPATH=/app`. With min instances 0 the service scales to zero, so the first request after idle is a cold start.

> **Container Registry is deprecated.** The build pushes to `gcr.io` and the script enables `containerregistry.googleapis.com`. Google is retiring Container Registry in favor of Artifact Registry (`REGION-docker.pkg.dev/PROJECT/REPOSITORY/IMAGE`); migrating `deploy/cloudbuild.yaml` and `deploy/deploy_gcloud.sh` is an open follow-up.

## The production image (`Dockerfile.production`)

Three stages:

1. **Python deps** (`python:3.10-slim`, the Hyak Spark env's Python): `uv pip install -r requirements.txt -c constraints.txt`.
2. **Frontend** (`node:22-slim`): `npm ci`, `npm run build` → `frontend/dist`.
3. **Runtime** (`python:3.10-slim`): copies `backend/`, `forecasting/`, `data/`, `config/` (needed: `config.py` loads `config/tuned_hyperparameters.json` at import), `config.py`, `results/chronological`, `results/multi_seed` and `results/conformal_band` (read by `GET /api/metrics/summary`, the track record and the error band), `store/`, and the built frontend. The build fails early if `store/` or `frontend/dist` is missing, `import fastapi, granian, deltalake, config` fails, or the store is not complete and current for the image (skipped with `--build-arg REQUIRE_CURRENT_STORE=0`, which CI's stub store uses). Runs as a non-root user on port 8080 with a `/health` healthcheck; starts Granian (`DATECT_USE_GRANIAN=1`, the default) or Uvicorn.

Build inputs are whitelisted: `.dockerignore` (Docker build context) and `.gcloudignore` (Cloud Build upload) admit only what the Dockerfile copies. A new directory the image needs must be added to the Dockerfile **and** both ignore files.

Run the image locally:

```bash
docker build -f Dockerfile.production -t datect .
docker run -p 8080:8080 datect          # http://localhost:8080
```

**CI** (`.github/workflows/docker.yml`): on pushes and pull requests to `main` (except docs/papers-only changes) it builds the image with a stub `store/` and fails unless `/health` answers within about a minute. It then prints the image's config hash next to the Hyak env's (`results/store/hyak_config_hash.txt`), builds a two-entry forecasts table inside the image (`python -m backend.precomputed sample`), and checks that the container serves that forecast from the store and computes another on demand.

## Environment variables

Read by the server code (`backend/`), `run_datect.py` and the image:

| Variable | Read by | Effect |
|---|---|---|
| `DATECT_PRECOMPUTED_DIR` | `backend/precomputed.py`, `run_datect.py` | Directory of the store's Delta tables. Unset → every request is computed on demand. `run_datect.py` sets it to `<repo>/store` when that exists; the image sets `/app/store`. |
| `ALLOWED_ORIGINS` | `backend/api.py` | Comma-separated CORS origins. Unset → `http://localhost:3000`, `http://localhost:5173`. Not needed for the single-origin Cloud Run deploy; set it if another origin calls the API. |
| `PORT` | Docker `CMD`, `backend/api.py` | Listen port: 8080 in the image; `python -m backend.api` defaults to 8000. |
| `DATECT_USE_GRANIAN` | Docker `CMD`, `backend/api.py` | `1` (default) Granian, otherwise Uvicorn. |
| `WORKERS` | `backend/api.py` | Granian worker count, only when started with `python -m backend.api` (default 1). The Docker `CMD` does not read it. |
| `VITE_API_URL` | `frontend/src/services/api.js` (build time) | API base URL baked into the frontend. Unset → same origin (`/api`); the Vite dev server proxies `/api` to `localhost:8000`. |

The engine's `DATECT_*` overrides (see `config.py`) apply to the API process too, and are part of the config hash; the deployed service uses none.

There is no runtime configuration endpoint: `POST /api/config` was removed, and task and model now travel with each request (`/api/retrospective` takes them in the body). `GET /api/config` remains read-only.

## Verify and operate

```bash
curl https://SERVICE_URL/health            # includes the store's tables, entries and stale rows
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
| `./store is missing or empty` | Build the store on Hyak and copy it (sections 1–2). |
| Docker build fails at `ls -la ./store/` | `store/` absent from the build context. |
| Container exits at startup | Check the logs; startup errors reach them (no stderr redirection). A missing `COPY` (e.g. `config/`) shows up as an import error. |
| `/api/metrics/summary` returns 404 | `results/chronological` missing from the image (without `results/multi_seed` the multi-seed fields are empty). |
| Forecasts, retrospective or spectral views are slow | `/health` shows why: no store, or its rows are stale (`stale_rows` > 0: built from other code, config, data or library versions). Rebuild the store at the deployed commit. Forecasts are precomputed for weekly dates and the `ensemble` model only; other requests are always computed. |

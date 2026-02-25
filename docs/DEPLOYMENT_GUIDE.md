# DATect Deployment Guide

Complete workflow for updating the pre-computed cache on Hyak and deploying to Google Cloud Run.

---

## Overview

```
Hyak (compute cluster)          Local Mac               Google Cloud
──────────────────────          ─────────────           ─────────────
Run precompute_cache.py  ──→   rsync cache down  ──→   ./deploy_gcloud.sh
  (~45–90 min)                  (2–5 min)                (~15–20 min build)
```

The cache is **baked into the Docker image** at deploy time. If you skip the rsync, GCP will deploy with a stale or empty cache and all retrospective/spectral endpoints will be slow or broken.

---

## Prerequisites (one-time setup)

### Local Mac: SSH config for Hyak
`~/.ssh/config` must contain:
```
Host klone-login
    User YOUR_UWNETID
    Hostname klone.hyak.uw.edu
    ServerAliveInterval 30
    ServerAliveCountMax 1200
    ControlMaster auto
    ControlPersist 3600
    ControlPath ~/.ssh/%r@klone-login:%p
```

### Local Mac: GCP tools installed
```bash
# Install gcloud CLI if not already installed
brew install --cask google-cloud-sdk

# Authenticate
gcloud auth login
gcloud auth application-default login
```

---

## Phase 1 — Run Precompute on Hyak

> Only needed when: model code changed, new data is available, or cache is missing/stale.
> Skip to Phase 2 if cache is already up to date.

```bash
# 1. SSH into the login node
ssh klone-login

# 2. Request a compute node (wait for "Granted job allocation" message)
salloc --account=stf --partition=cpu-g2 --nodes=1 --ntasks=1 \
  --cpus-per-task=192 --mem=500G --time=4:00:00 --job-name=datect-precompute

# 3. Activate environment
source /sw/contrib/foster-src/python/miniconda/3.8/etc/profile.d/conda.sh
conda activate /gscratch/stf/ac283/envs/datect_scratch

# 4. Pull latest code
cd /gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid
git checkout main
git pull origin main

# 5. Install/update dependencies
pip install -r requirements.txt --no-cache-dir

# 6. Run precompute — takes 45–90 min, watch for all 8 combinations to complete
python precompute_cache.py
```

**Expected output** — you should see all 8 of these complete:
```
  regression + ensemble...    ✓
  regression + xgboost...     ✓
  regression + rf...          ✓
  regression + naive...       ✓
  regression + linear...      ✓
  classification + ensemble...✓
  classification + naive...   ✓
  classification + logistic...✓
```

### Verify cache is complete before syncing
```bash
# Still on Hyak — confirm all files exist
ls /gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid/cache/retrospective/ | wc -l
# Must show: 8

ls /gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid/cache/
# Must show: manifest.json  retrospective/  spectral/  visualizations/
```

### Clean up Hyak resources when done
```bash
exit                                  # leave the compute node
scancel --name datect-precompute      # release the allocation
```

---

## Phase 2 — Sync Cache to Local Mac

> Run this from a **local terminal** (not on Hyak). `/gscratch` is accessible from the
> login node so no compute node required — no SSH config update needed.

```bash
# From project root on your Mac:
cd /Users/ansonchen/Downloads/GitHub/DATect-Forecasting-Domoic-Acid

# Sync cache down (only transfers changed files on re-runs)
rsync -avz klone-login:/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid/cache/ ./cache/

# Verify local cache looks correct
ls -la cache/
# Expected:
#   cache/manifest.json
#   cache/retrospective/    ← 8 .json files
#   cache/spectral/         ← per-site .json files
#   cache/visualizations/   ← correlation heatmap .json files
```

---

## Phase 3 — Deploy to Google Cloud

```bash
# Set your GCP project ID (find it at console.cloud.google.com)
export PROJECT_ID=your-gcp-project-id
export REGION=us-west1        # or your preferred region

# Run deploy script from project root
cd /Users/ansonchen/Downloads/GitHub/DATect-Forecasting-Domoic-Acid
./deploy_gcloud.sh
```

**When prompted "Re-generate cache? (y/N)"** → type **N**
The Hyak-generated cache is correct; do not regenerate locally.

### What deploy_gcloud.sh does
1. Builds the React frontend locally (`build_frontend.sh`)
2. Uploads source + cache to Cloud Build (~1–2 min, .venv excluded)
3. Builds Docker image in Cloud Build (~10–15 min)
4. Pushes image to Google Container Registry
5. Deploys to Cloud Run with cost-optimized settings
6. Runs a health check on the live URL

### Cloud Run configuration (cost-optimized)
| Setting | Value | Reason |
|---------|-------|--------|
| `--min-instances` | `0` | Scales to zero when idle → $0 cost between demos |
| `--max-instances` | `3` | Enough for demo traffic |
| `--concurrency` | `20` | Realistic for demo load |
| `--memory` | `1Gi` | Required for ML models + parquet data in memory |
| `--cpu` | `1` | Sufficient for cached-response serving |
| `--timeout` | `300s` | Allows live forecasts to complete |

### Expected cost
- **Idle (no traffic):** ~$0/month — scales to zero
- **Active demo:** stays within GCP free tier (360k GB-seconds/month free)
- **Cold start:** first request after idle takes ~10–15s (Python + models loading)

### Warm up before a demo
```bash
# Hit health endpoint 30 seconds before showing to someone
curl https://YOUR_SERVICE_URL/health
curl https://YOUR_SERVICE_URL/api/cache
```

---

## Phase 4 — Verify Deployment

```bash
# Basic health check
curl https://YOUR_SERVICE_URL/health

# Confirm cache is loaded
curl https://YOUR_SERVICE_URL/api/cache

# View live logs if something seems wrong
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=datect-api" \
  --limit 50 --project $PROJECT_ID
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `rsync: connection refused` | SSH config issue | Confirm `klone-login` works: `ssh klone-login echo ok` |
| Cache missing `retrospective/` after rsync | Precompute didn't finish or failed | Re-run Phase 1 and check for errors in output |
| Docker build fails: `ls -la ./cache/` error | Cache wasn't synced before deploy | Re-run Phase 2 then re-deploy |
| Deploy script tries to run precompute locally | Cache dir missing or empty | Complete Phase 2 first |
| GCP build timeout | Build took >20 min | Re-run `./deploy_gcloud.sh` — Cloud Build caches layers |
| Cold start takes >30s | First request loads all ML models + 500MB parquet | Normal — warm up before demo (see above) |
| `Project ID not set` error | Missing env var | `export PROJECT_ID=your-project-id` |
| `gcloud: command not found` | gcloud CLI not installed | `brew install --cask google-cloud-sdk` |

---

## Cost Monitoring

Set a budget alert so you're never surprised:
1. Go to [console.cloud.google.com](https://console.cloud.google.com) → Billing → Budgets & alerts
2. Create budget: **$5/month** with email alert at 50% and 100%

Check for accidentally running resources:
```bash
# VMs (most common surprise charge ~$3–5/day each)
gcloud compute instances list --project $PROJECT_ID

# All Cloud Run services
gcloud run services list --platform=managed --project $PROJECT_ID

# Delete service when not needed for extended periods
gcloud run services delete datect-api --platform managed --region $REGION
```

---

## Quick Reference

```bash
# Full workflow in one block:
ssh klone-login "cd /gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid && \
  ls cache/retrospective/ | wc -l"                              # verify cache (should be 8)

cd /Users/ansonchen/Downloads/GitHub/DATect-Forecasting-Domoic-Acid
rsync -avz klone-login:/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid/cache/ ./cache/

export PROJECT_ID=your-gcp-project-id
./deploy_gcloud.sh                                              # answer N to cache regeneration
```

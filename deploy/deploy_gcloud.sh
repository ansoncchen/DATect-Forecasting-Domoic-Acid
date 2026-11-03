#!/bin/bash
# DATect Google Cloud Deployment
# Ensures local ./store exists, then builds the image (frontend included) with Cloud Build and deploys to Cloud Run

set -e

# Run from the repo root (this script lives in deploy/)
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

if [ -z "$PROJECT_ID" ]; then
    print_error "PROJECT_ID environment variable is required"
    echo "Run: export PROJECT_ID=datect-forecasting-domoic-acid"
    exit 1
fi

if [ -z "$REGION" ]; then
    export REGION="us-west1"
    print_warning "REGION not set, using default: $REGION"
fi

print_status "🚀 Starting DATect Google Cloud Deployment"
echo "=================================================="
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "=================================================="

# The precomputed store (Delta tables) is built on Hyak by datect_spark/store.sbatch,
# which saves it to ~/datect-store/<commit>/; copy that here. The image bakes it in.
# Build it at the commit you deploy: a store from other code, config or data is
# ignored (config hash) and every request is computed on demand.
if [ ! -d "./store" ] || [ -z "$(ls -A ./store)" ]; then
    print_error "./store is missing or empty. Build it on Hyak (datect_spark/store.sbatch)"
    echo "and copy it here, e.g.: rsync -a klone-login:datect-store/\$(git rev-parse HEAD)/ ./store/"
    exit 1
fi

STORE_SIZE=$(du -sh ./store | cut -f1)
STORE_TABLES=$(ls ./store | tr '\n' ' ')
print_success "Store ready: $STORE_SIZE ($STORE_TABLES)"

print_status "🔧 Configuring Google Cloud..."
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION

print_status "🔌 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

print_status "🏗️  Building and deploying with Cloud Build..."
gcloud builds submit \
    --config=deploy/cloudbuild.yaml \
    --substitutions=_REGION=$REGION \
    --timeout=2400s

print_status "🌐 Getting service URL..."
SERVICE_URL=$(gcloud run services describe datect-api --platform=managed --region=$REGION --format="value(status.url)")

if [ -z "$SERVICE_URL" ]; then
    print_error "Failed to get service URL"
    exit 1
fi

print_status "🧪 Testing deployment..."
HEALTH_CHECK="${SERVICE_URL}/health"
if curl -sf "$HEALTH_CHECK" > /dev/null; then
    print_success "Health check passed"
else
    print_warning "Health check failed, but deployment may still be starting up"
fi

echo
print_success "🎉 DATect deployed successfully!"
echo "=================================================="
echo -e "${GREEN}🔗 Application URL: $SERVICE_URL${NC}"
echo -e "${GREEN}🔗 API Documentation: $SERVICE_URL/docs${NC}"
echo -e "${GREEN}💚 Health: $SERVICE_URL/health${NC}"
echo "=================================================="

read -p "Open in browser? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v open >/dev/null 2>&1; then
        open "$SERVICE_URL"
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$SERVICE_URL"
    else
        print_warning "Cannot open browser automatically"
    fi
fi

echo
print_status "📚 Useful commands:"
echo "View logs: gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=datect-api\" --limit 50 --format json"
echo "Update service: gcloud run deploy datect-api --image gcr.io/$PROJECT_ID/datect:latest --platform managed --region $REGION"
echo "Delete service: gcloud run services delete datect-api --platform managed --region $REGION"
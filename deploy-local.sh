#!/bin/bash
########################################
# 
# run chmod +x deploy.sh to make the script executable
# 
# Execute this script:  src/deploy.sh
#
########################################

set -e

LOCAL_IMAGE_NAME="por-local-image" 
GCP_IMAGE_NAME="por-image"
GCP_PROJECT_ID="pim-87183"
GCP_REGION_NAME="us-central1"
GCP_REPO_NAME="pim-online-retail"

echo "Deploying to Docker Container"

# Build a local image
docker build -t $LOCAL_IMAGE_NAME .

# Run image
docker run -p 8501:8501 $LOCAL_IMAGE_NAME

# Tag the local image
# docker tag $LOCAL_IMAGE_NAME:latest $GCP_REGION_NAME-docker.pkg.dev/$GCP_PROJECT_ID/$GCP_REPO_NAME/$GCP_IMAGE_NAME

# Push to gcr
# docker push gcr.io/$GCP_PROJECT_ID/$LOCAL_IMAGE_NAME

# Push to GCP Artifact Registry
# docker push $GCP_REGION_NAME-docker.pkg.dev/$GCP_PROJECT_ID/$GCP_REPO_NAME/$GCP_IMAGE_NAME
# sudo ../../../Downloads/google-cloud-sdk/bin/gcloud builds submit --region=$GCP_REGION_NAME --tag $GCP_REGION_NAME-docker.pkg.dev/$GCP_PROJECT_ID/$GCP_REPO_NAME/$GCP_IMAGE_NAME:latest
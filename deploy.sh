#!/bin/bash
########################################
# 
# run chmod +x deploy.sh to make the script executable
# 
# Execute this script:  src/deploy.sh
#
########################################

set -e

# Set environment variables
LOCAL_IMAGE_NAME="por-local-image" 
GCP_PROJECT_ID="pim-85335"
GCP_REGION_NAME="us-central1"
GCP_REPO_NAME="pim-online-retail"
GCP_IMAGE_NAME="por-image"
TAG_NAME="latest"

# Function to check if repository exists
check_and_create_repo() {
    echo "Checking if repository $GCP_REPO_NAME exists..."
    
    if ! gcloud artifacts repositories describe $GCP_REPO_NAME \
        --project=$GCP_PROJECT_ID \
        --location=$GCP_REGION_NAME &>/dev/null; then
        echo "Repository $GCP_REPO_NAME does not exist. Creating it..."
        
        gcloud artifacts repositories create $GCP_REPO_NAME \
            --project=$GCP_PROJECT_ID \
            --repository-format=docker \
            --location=$GCP_REGION_NAME \
            --description="Docker repository"
        
        if [ $? -eq 0 ]; then
            echo "Repository $GCP_REPO_NAME created successfully."
        else
            echo "Failed to create repository $GCP_REPO_NAME. Exiting."
            exit 1
        fi
    else
        echo "Repository $GCP_REPO_NAME already exists."
    fi
}

# Call the function to check and create repository
check_and_create_repo


echo "Deploying to Docker Container"

# Build a local image
echo "Building local Docker image..."
docker build -t $LOCAL_IMAGE_NAME .

# Set the image URI
IMAGE_URI="$GCP_REGION_NAME-docker.pkg.dev/$GCP_PROJECT_ID/$GCP_REPO_NAME/$GCP_IMAGE_NAME:$TAG_NAME"

# Submit the build to Artifact Registry
echo "Submitting build to Artifact Registry..."
gcloud builds submit --region=$GCP_REGION_NAME --tag $IMAGE_URI
# docker push $GCP_REGION_NAME-docker.pkg.dev/$GCP_PROJECT_ID/$GCP_REPO_NAME/$GCP_IMAGE_NAME

# Set the Cloud Run region
echo "Setting Cloud Run region..."
gcloud config set run/region $GCP_REGION_NAME

# Deploy the service to Cloud Run
echo "Deploying $GCP_IMAGE_NAME to Cloud Run..."
gcloud run deploy $GCP_IMAGE_NAME \
    --image $IMAGE_URI \
    --memory 256Mi \
    --cpu 1 \
    --max-instances 1 \
    --allow-unauthenticated
    # --set-env-vars "ENV_VAR1=value1,ENV_VAR2=value2" \
    # --min-instances: Set the minimum number of instances.
    # --timeout: Set the request timeout.
    # --concurrency: Set the maximum number of concurrent requests per instance.
    
echo "Deployment complete."

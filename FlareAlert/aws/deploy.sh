#!/bin/bash

set -e

ENVIRONMENT=${1:-dev}
REGION=${2:-us-east-1}
STACK_NAME="flarealert-${ENVIRONMENT}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO_NAME="${ENVIRONMENT}-flarealert-api"

echo "Deploying FlareAlert to ${ENVIRONMENT} environment in ${REGION}..."

if [ -z "$DATABASE_PASSWORD" ]; then
    echo "Error: DATABASE_PASSWORD environment variable is required"
    exit 1
fi

echo "Setting up ECR repository..."

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --region ${REGION} 2>/dev/null || \
aws ecr create-repository --repository-name ${ECR_REPO_NAME} --region ${REGION}

# Get ECR login token
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

echo "Building and pushing Docker image..."

cd ../backend

# Build the Lambda container image
docker build -f Dockerfile.lambda -t ${ECR_REPO_NAME}:latest .

# Tag and push to ECR
docker tag ${ECR_REPO_NAME}:latest ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}:latest
docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}:latest

# Get the image URI
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}:latest"

echo "Deploying CloudFormation stack..."

aws cloudformation deploy \
    --template-file ../aws/cloudformation.yaml \
    --stack-name ${STACK_NAME} \
    --parameter-overrides \
        Environment=${ENVIRONMENT} \
        DatabasePassword=${DATABASE_PASSWORD} \
        ImageUri=${IMAGE_URI} \
    --capabilities CAPABILITY_NAMED_IAM \
    --region ${REGION}

echo "Getting stack outputs..."

API_URL=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME} \
    --region ${REGION} \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
    --output text)

DB_ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME} \
    --region ${REGION} \
    --query 'Stacks[0].Outputs[?OutputKey==`DatabaseEndpoint`].OutputValue' \
    --output text)

BUCKET_NAME=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME} \
    --region ${REGION} \
    --query 'Stacks[0].Outputs[?OutputKey==`ModelBucketName`].OutputValue' \
    --output text)

echo "Deployment complete!"
echo "API URL: ${API_URL}"
echo "Database Endpoint: ${DB_ENDPOINT}"
echo "Model Bucket: ${BUCKET_NAME}"

echo "Deployment script completed successfully!"

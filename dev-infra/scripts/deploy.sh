#!/bin/bash
# One-command deployment script

set -e

echo "🚀 Deploying OCR Pipeline to AWS"

# Build Docker image
echo "🏗️ Building Docker image..."
docker build -f dev-infra/Dockerfile -t ocr-pipeline .

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names ocr-pipeline || \
aws ecr create-repository --repository-name ocr-pipeline

# Get ECR login and push image
echo "📤 Pushing to ECR..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-us-east-1}
ECR_URI=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/ocr-pipeline

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI
docker tag ocr-pipeline:latest $ECR_URI:latest
docker push $ECR_URI:latest

# Update job definition with actual image URI
echo "📝 Updating job definition..."
aws batch register-job-definition \
    --job-definition-name ocr-processing-job \
    --type container \
    --container-properties "$(cat <<EOF
{
    "image": "$ECR_URI:latest",
    "vcpus": 2,
    "memory": 4096,
    "jobRoleArn": "arn:aws:iam::${ACCOUNT_ID}:role/OCRBatchJobRole",
    "executionRoleArn": "arn:aws:iam::${ACCOUNT_ID}:role/ecsTaskExecutionRole",
    "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
            "awslogs-group": "/aws/batch/ocr-processing",
            "awslogs-region": "$REGION"
        }
    }
}
EOF
)"

echo "✅ Deployment complete!"
echo "ECR Image: $ECR_URI:latest"
echo "Ready to submit jobs!"

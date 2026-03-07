#!/bin/bash

# AWS Batch setup script for OCR pipeline
# Run this in AWS CloudShell or with AWS CLI configured

set -e

# Configuration
REGION=${AWS_REGION:-us-east-1}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET_NAME=${S3_BUCKET_NAME:-tokenworks-ocr-$(date +%s)}

echo "🚀 Setting up AWS infrastructure for OCR Pipeline"
echo "Region: $REGION"
echo "Account: $ACCOUNT_ID"
echo "S3 Bucket: $BUCKET_NAME"

# Create S3 bucket
echo "📦 Creating S3 bucket..."
aws s3 mb s3://$BUCKET_NAME --region $REGION

# Create S3 folders
echo "📁 Creating S3 folder structure..."
aws s3api put-object --bucket $BUCKET_NAME --key failed/
aws s3api put-object --bucket $BUCKET_NAME --key heavy/
aws s3api put-object --bucket $BUCKET_NAME --key results/
aws s3api put-object --bucket $BUCKET_NAME --key processing/
aws s3api put-object --bucket $BUCKET_NAME --key manual_review/
aws s3api put-object --bucket $BUCKET_NAME --key supervisor_logs/

# Create IAM role for Batch jobs
echo "🔐 Creating IAM role for Batch jobs..."
cat > batch-job-role-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::$BUCKET_NAME",
                "arn:aws:s3:::$BUCKET_NAME/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogStreams"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        }
    ]
}
EOF

aws iam create-role --role-name OCRBatchJobRole \
    --assume-role-policy-document '{"Version": "2012-10-17","Statement": [{ "Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'

aws iam put-role-policy --role-name OCRBatchJobRole \
    --policy-name OCRBatchJobPolicy \
    --policy-document file://batch-job-role-policy.json

# Create compute environment
echo "🖥️ Creating compute environment..."
aws batch create-compute-environment \
    --compute-environment-name ocr-compute-env \
    --type MANAGED \
    --state ENABLED \
    --compute-resources "type=EC2,allocationStrategy=BEST_FIT_PROGRESSIVE,minvCpus=0,maxvCpus=16,desiredvCpus=2,instanceTypes=t3.medium,t3.large,m5.large,m5.xlarge, subnets=subnet-12345,securityGroupIds=sg-12345,instanceRole=ecsInstanceRole"

# Create job queues
echo "📋 Creating standard job queue..."
aws batch create-job-queue \
    --job-queue-name ocr-standard-queue \
    --state ENABLED \
    --priority 1 \
    --compute-environment-order "order=1,computeEnvironment=ocr-compute-env"

echo "💪 Creating high-memory job queue..."
aws batch create-job-queue \
    --job-queue-name ocr-high-memory-queue \
    --state ENABLED \
    --priority 2 \
    --compute-environment-order "order=1,computeEnvironment=ocr-compute-env"

# Create job definition (will be updated with ECR image URI after build)
echo "📝 Creating job definition placeholder..."
cat > job-definition.json << EOF
{
    "jobDefinitionName": "ocr-processing-job",
    "type": "container",
    "containerProperties": {
        "image": "PLACEHOLDER_ECR_URI",
        "vcpus": 2,
        "memory": 4096,
        "jobRoleArn": "arn:aws:iam::$ACCOUNT_ID:role/OCRBatchJobRole",
        "executionRoleArn": "arn:aws:iam::$ACCOUNT_ID:role/ecsTaskExecutionRole",
        "logConfiguration": {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group": "/aws/batch/ocr-processing",
                "awslogs-region": "$REGION"
            }
        }
    }
}
EOF

aws batch register-job-definition --cli-input-json file://job-definition.json

# Create CloudWatch log group
echo "📊 Creating CloudWatch log group..."
aws logs create-log-group --log-group-name /aws/batch/ocr-processing

echo "✅ AWS infrastructure setup complete!"
echo ""
echo "📋 Summary:"
echo "S3 Bucket: $BUCKET_NAME"
echo "Standard Queue: ocr-standard-queue"
echo "High Memory Queue: ocr-high-memory-queue"
echo "Job Definition: ocr-processing-job"
echo ""
echo "🔧 Next steps:"
echo "1. Build and push Docker image to ECR"
echo "2. Update job definition with ECR image URI"
echo "3. Set environment variables in your deployment"
echo "4. Test with a sample job"
echo "5. Set up supervisor Lambda function for periodic escalation checking"

# Cleanup
rm -f batch-job-role-policy.json job-definition.json

#!/bin/bash
# Getting Started Script for OCR Pipeline AWS Deployment
# Run this to get a complete walkthrough

echo "🚀 OCR Pipeline AWS Deployment - Getting Started"
echo "==============================================="
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install AWS CLI v2 first:"
    echo "   https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop first:"
    echo "   https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✅ Prerequisites OK"
echo ""

# Make scripts executable
echo "🔧 Setting up scripts..."
chmod +x infrastructure/aws_batch_setup.sh scripts/deploy.sh scripts/submit_job.sh
echo "✅ Scripts ready"
echo ""

# Interactive setup
echo "🔄 Step-by-step AWS deployment:"
echo ""

echo "Step 1: Configure AWS CLI (if not done already)"
echo "   aws configure"
echo "   # Enter your AWS Access Key, Secret Key, default region (us-east-1)"
echo ""

read -p "Have you configured AWS CLI? (y/n): " aws_configured
if [[ $aws_configured != "y" ]]; then
    echo "Please run 'aws configure' first, then rerun this script."
    exit 1
fi

echo ""
echo "Step 2: Set up AWS infrastructure"
echo "   This will create S3 bucket, IAM roles, Batch queues, etc."
read -p "Ready to create AWS infrastructure? (y/n): " create_infra

if [[ $create_infra == "y" ]]; then
    echo "🏗️ Creating AWS infrastructure..."
    ./infrastructure/aws_batch_setup.sh

    if [[ $? -eq 0 ]]; then
        echo "✅ AWS infrastructure created successfully!"
    else
        echo "❌ AWS infrastructure setup failed. Check the error messages above."
        exit 1
    fi
else
    echo "Skipping infrastructure setup. Make sure you run './infrastructure/aws_batch_setup.sh' manually."
fi

echo ""
echo "Step 3: Configure environment variables"
echo "   Copy the example file and edit it:"
echo "   cp env.example .env"
echo "   # Then edit .env with your actual values"
echo ""

read -p "Have you configured your .env file? (y/n): " env_configured
if [[ $env_configured != "y" ]]; then
    echo "Please copy and configure your .env file, then continue."
    echo "You can use default values for testing, but update S3_BUCKET with the one created above."
fi

echo ""
echo "Step 4: Build and deploy Docker image"
echo "   This builds your container and pushes to Amazon ECR"
read -p "Ready to build and deploy? (y/n): " deploy_ready

if [[ $deploy_ready == "y" ]]; then
    echo "🏗️ Building and deploying..."
    ./scripts/deploy.sh

    if [[ $? -eq 0 ]]; then
        echo "✅ Deployment successful!"
    else
        echo "❌ Deployment failed. Check Docker and AWS permissions."
        exit 1
    fi
else
    echo "Skipping deployment. Run './scripts/deploy.sh' manually when ready."
fi

echo ""
echo "🎉 Setup complete! Next steps:"
echo ""
echo "1. Upload a test file:"
echo "   aws s3 cp ../data/samples/test.png s3://YOUR_BUCKET/input/"
echo ""
echo "2. Submit a job:"
echo "   ./scripts/submit_job.sh s3://YOUR_BUCKET/input/test.png"
echo ""
echo "3. Check job status:"
echo "   aws batch list-jobs --job-queue ocr-standard-queue"
echo ""
echo "4. Monitor results:"
echo "   aws s3 ls s3://YOUR_BUCKET/results/ --recursive"
echo ""
echo "📖 For detailed documentation, see README.md"
echo "🆘 For troubleshooting, check the README troubleshooting section"

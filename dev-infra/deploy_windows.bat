@echo off
REM Windows Batch Script for OCR Pipeline AWS Deployment
REM Run this as Administrator in Command Prompt

echo 🚀 OCR Pipeline AWS Deployment - Windows Edition
echo =================================================
echo.

echo 📋 Checking prerequisites...

REM Check AWS CLI
aws --version >nul 2>&1
if errorlevel 1 (
    echo ❌ AWS CLI not found. Please install AWS CLI v2 first:
    echo    https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
    pause
    exit /b 1
) else (
    echo ✅ AWS CLI found
)

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker not found. Please install Docker Desktop first:
    echo    https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
) else (
    echo ✅ Docker found
)

echo.
echo 🔧 Setting up environment...

REM Check if we're in the right directory
if not exist "cloud\config.py" (
    echo ❌ Please run this script from the dev-infra directory
    echo    cd C:\path\to\your\ocr-pipeline\dev-infra
    pause
    exit /b 1
)

echo ✅ Environment ready
echo.

echo Step 1: Configure AWS CLI (if not done already)
echo    Run: aws configure
echo    Enter your AWS Access Key, Secret Key, default region (us-east-1)
echo.

set /p aws_configured="Have you configured AWS CLI? (y/n): "
if not "%aws_configured%"=="y" (
    echo Please run 'aws configure' first, then rerun this script.
    pause
    exit /b 1
)

echo.
echo Step 2: Set up AWS infrastructure
echo    This will create S3 bucket, IAM roles, Batch queues, etc.
set /p create_infra="Ready to create AWS infrastructure? (y/n): "

if "%create_infra%"=="y" (
    echo 🏗️ Creating AWS infrastructure...
    echo.
    echo ⚠️  For Windows, we recommend using AWS CloudShell or WSL.
    echo    Please open AWS CloudShell in your browser and run:
    echo.
    echo    git clone https://github.com/your-username/ocr-pipeline.git
    echo    cd ocr-pipeline/dev-infra
    echo    ./infrastructure/aws_batch_setup.sh
    echo.
    echo    Then return here and continue with Step 3.
    echo.
    pause
) else (
    echo Skipping infrastructure setup. Make sure you run the setup manually.
)

echo.
echo Step 3: Configure environment variables
echo    Copy the example file and edit it:
echo    copy env.example .env
echo    # Then edit .env with your actual values
echo.

set /p env_configured="Have you configured your .env file? (y/n): "
if not "%env_configured%"=="y" (
    echo Please copy and configure your .env file, then continue.
    echo You can use default values for testing, but update S3_BUCKET with the one created above.
    echo.
    copy env.example .env 2>nul
    echo Created .env file. Please edit it now with your actual values.
    notepad .env
)

echo.
echo Step 4: Build and deploy Docker image
echo    This builds your container and pushes to Amazon ECR
set /p deploy_ready="Ready to build and deploy? (y/n): "

if "%deploy_ready%"=="y" (
    echo 🏗️ Building and deploying...
    echo.
    echo ⚠️  For Windows, we recommend using AWS CloudShell or WSL.
    echo    In CloudShell, run: ./scripts/deploy.sh
    echo.
    echo    Then return here and continue with testing.
    echo.
    pause
) else (
    echo Skipping deployment. Run deployment manually when ready.
)

echo.
echo 🎉 Setup complete! Next steps:
echo.
echo 1. Upload a test file:
echo    aws s3 cp ..\data\samples\test.png s3://YOUR_BUCKET/input/
echo.
echo 2. Submit a job:
echo    # In CloudShell: ./scripts/submit_job.sh s3://YOUR_BUCKET/input/test.png
echo.
echo 3. Check job status:
echo    aws batch list-jobs --job-queue ocr-standard-queue
echo.
echo 4. Monitor results:
echo    aws s3 ls s3://YOUR_BUCKET/results/ --recursive
echo.
echo 📖 For detailed documentation, see README.md
echo 🆘 For troubleshooting, check the README troubleshooting section
echo.
pause

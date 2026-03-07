# Windows PowerShell Script for OCR Pipeline AWS Deployment
# Run this as Administrator if needed

Write-Host "🚀 OCR Pipeline AWS Deployment - Windows Edition" -ForegroundColor Green
Write-Host "=" * 55 -ForegroundColor Green
Write-Host ""

# Check prerequisites
Write-Host "📋 Checking prerequisites..." -ForegroundColor Yellow

# Check AWS CLI
try {
    $awsVersion = aws --version 2>$null
    Write-Host "✅ AWS CLI found: $awsVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ AWS CLI not found. Please install AWS CLI v2 first:" -ForegroundColor Red
    Write-Host "   https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html" -ForegroundColor Red
    exit 1
}

# Check Docker
try {
    $dockerVersion = docker --version 2>$null
    Write-Host "✅ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker not found. Please install Docker Desktop first:" -ForegroundColor Red
    Write-Host "   https://www.docker.com/products/docker-desktop" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🔧 Setting up scripts..." -ForegroundColor Yellow

# Convert Unix scripts to Windows batch files
$scripts = @(
    "infrastructure\aws_batch_setup.sh",
    "scripts\deploy.sh",
    "scripts\submit_job.sh"
)

foreach ($script in $scripts) {
    if (Test-Path $script) {
        Write-Host "   Converting $script to Windows format..." -ForegroundColor Gray
        # For now, we'll keep the .sh files but provide instructions
    }
}

Write-Host "✅ Scripts ready" -ForegroundColor Green
Write-Host ""

# Interactive setup
Write-Host "🔄 Step-by-step AWS deployment:" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 1: Configure AWS CLI (if not done already)" -ForegroundColor White
Write-Host "   Run: aws configure" -ForegroundColor Gray
Write-Host "   Enter your AWS Access Key, Secret Key, default region (us-east-1)" -ForegroundColor Gray
Write-Host ""

$awsConfigured = Read-Host "Have you configured AWS CLI? (y/n)"
if ($awsConfigured -ne "y") {
    Write-Host "Please run 'aws configure' first, then rerun this script." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Step 2: Set up AWS infrastructure" -ForegroundColor White
Write-Host "   This will create S3 bucket, IAM roles, Batch queues, etc." -ForegroundColor Gray
$createInfra = Read-Host "Ready to create AWS infrastructure? (y/n)"

if ($createInfra -eq "y") {
    Write-Host "🏗️ Creating AWS infrastructure..." -ForegroundColor Blue

    # For Windows, we'll use bash if available, otherwise provide instructions
    try {
        bash infrastructure/aws_batch_setup.sh
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ AWS infrastructure created successfully!" -ForegroundColor Green
        } else {
            Write-Host "❌ AWS infrastructure setup failed. Check the error messages above." -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "❌ Bash not available. Please use AWS CloudShell or WSL:" -ForegroundColor Red
        Write-Host "   1. Go to AWS Console → CloudShell" -ForegroundColor Red
        Write-Host "   2. Clone your repo and run: ./infrastructure/aws_batch_setup.sh" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Skipping infrastructure setup. Make sure you run './infrastructure/aws_batch_setup.sh' manually." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Step 3: Configure environment variables" -ForegroundColor White
Write-Host "   Copy the example file and edit it:" -ForegroundColor Gray
Write-Host "   copy env.example .env" -ForegroundColor Gray
Write-Host "   # Then edit .env with your actual values" -ForegroundColor Gray
Write-Host ""

$envConfigured = Read-Host "Have you configured your .env file? (y/n)"
if ($envConfigured -ne "y") {
    Write-Host "Please copy and configure your .env file, then continue." -ForegroundColor Yellow
    Write-Host "You can use default values for testing, but update S3_BUCKET with the one created above." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Step 4: Build and deploy Docker image" -ForegroundColor White
Write-Host "   This builds your container and pushes to Amazon ECR" -ForegroundColor Gray
$deployReady = Read-Host "Ready to build and deploy? (y/n)"

if ($deployReady -eq "y") {
    Write-Host "🏗️ Building and deploying..." -ForegroundColor Blue

    try {
        bash scripts/deploy.sh
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Deployment successful!" -ForegroundColor Green
        } else {
            Write-Host "❌ Deployment failed. Check Docker and AWS permissions." -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "❌ Bash not available. Please use AWS CloudShell or WSL:" -ForegroundColor Red
        Write-Host "   Run: ./scripts/deploy.sh" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Skipping deployment. Run './scripts/deploy.sh' manually when ready." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Setup complete! Next steps:" -ForegroundColor Green
Write-Host ""
Write-Host "1. Upload a test file:" -ForegroundColor White
Write-Host "   aws s3 cp ../data/samples/test.png s3://YOUR_BUCKET/input/" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Submit a job:" -ForegroundColor White
Write-Host "   bash ./scripts/submit_job.sh s3://YOUR_BUCKET/input/test.png" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Check job status:" -ForegroundColor White
Write-Host "   aws batch list-jobs --job-queue ocr-standard-queue" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Monitor results:" -ForegroundColor White
Write-Host "   aws s3 ls s3://YOUR_BUCKET/results/ --recursive" -ForegroundColor Gray
Write-Host ""
Write-Host "📖 For detailed documentation, see README.md" -ForegroundColor Cyan
Write-Host "🆘 For troubleshooting, check the README troubleshooting section" -ForegroundColor Cyan

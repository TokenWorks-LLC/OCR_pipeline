# AWS Cloud Infrastructure for OCR Pipeline

This directory contains all AWS deployment infrastructure, isolated from the main OCR codebase.

## 📁 Directory Structure

```
dev-infra/
├── cloud/                          # Core cloud orchestration code
│   ├── __init__.py
│   ├── config.py                   # Cloud configuration management
│   ├── worker/
│   │   └── job_processor.py        # Individual job processing
│   └── supervisor/
│       ├── escalation_manager.py   # Handles failed job escalation
│       └── supervisor_runner.py    # Supervisor execution script
├── infrastructure/                 # AWS infrastructure setup
│   ├── aws_batch_setup.sh          # One-command AWS setup
│   └── supervisor_lambda.py        # Lambda function for supervisor
├── scripts/                        # Deployment and management scripts
│   ├── deploy.sh                   # Build and deploy to ECR
│   └── submit_job.sh               # Submit jobs to AWS Batch
├── Dockerfile                      # Container definition
├── env.example                     # Environment variables template
└── README.md                       # This file
```

## 🚀 Quick Start for Beginners

### Step 1: Prerequisites

1. **AWS Account**: Make sure you have an AWS account
2. **AWS CLI**: Install AWS CLI v2 on your machine
3. **Docker**: Install Docker Desktop
4. **Git**: For version control

### Step 2: AWS Setup

#### Option A: Use AWS CloudShell (Recommended for Beginners)

1. Go to AWS Console → CloudShell
2. Clone your repository:
   ```bash
   git clone https://github.com/your-repo/ocr-pipeline.git
   cd ocr-pipeline/dev-infra
   ```

3. Run the setup script:
   ```bash
   chmod +x infrastructure/aws_batch_setup.sh
   ./infrastructure/aws_batch_setup.sh
   ```

#### Option B: Local AWS CLI Setup

1. Configure AWS CLI:
   ```bash
   aws configure
   # Enter your AWS Access Key, Secret Key, region (us-east-1), and output format (json)
   ```

2. Run setup:
   ```bash
   cd dev-infra
   chmod +x infrastructure/aws_batch_setup.sh scripts/deploy.sh scripts/submit_job.sh
   ./infrastructure/aws_batch_setup.sh
   ```

### Step 3: Build and Deploy

1. **Build and push Docker image**:
   ```bash
   ./scripts/deploy.sh
   ```

2. **Set environment variables** (copy and edit):
   ```bash
   cp env.example .env
   # Edit .env with your actual values
   ```

### Step 4: Test Your Setup

1. **Upload a test file to S3**:
   ```bash
   # Replace 'your-bucket-name' with actual bucket from setup
   aws s3 cp ../data/samples/test.png s3://your-bucket-name/input/
   ```

2. **Submit a job**:
   ```bash
   ./scripts/submit_job.sh s3://your-bucket-name/input/test.png
   ```

3. **Check job status**:
   ```bash
   aws batch list-jobs --job-queue ocr-standard-queue
   ```

## 🔍 Understanding the Architecture

### How Jobs Flow Through the System

1. **Job Submission**: Files are submitted to the standard queue
2. **Processing**: Worker nodes process files using your existing OCR pipeline
3. **Escalation Check**: Results are checked against thresholds:
   - Confidence < 80%
   - Documents > 50 pages
   - Processing time > 10 minutes
4. **Escalation**: Failed jobs go to `failed/` or `heavy/` S3 folders
5. **Supervisor**: Every 10 minutes, supervisor scans folders and resubmits to high-memory queue
6. **Manual Review**: After 3 retries, jobs go to manual review

### S3 Bucket Structure

```
your-bucket/
├── input/           # Upload your files here
├── results/         # Successful processing results
├── failed/          # Jobs needing escalation (low confidence, errors)
├── heavy/           # Large/slow jobs needing escalation
├── processing/      # Currently being reprocessed
├── manual_review/   # Requires human intervention
└── supervisor_logs/ # Supervisor execution reports
```

## 📊 Monitoring and Health Checks

### Check Queue Status
```bash
# See all jobs in queues
aws batch describe-job-queues --job-queues ocr-standard-queue ocr-high-memory-queue

# List recent jobs
aws batch list-jobs --job-queue ocr-standard-queue --max-results 10
```

### Check S3 Folders
```bash
# See what's in escalation folders
aws s3 ls s3://your-bucket/failed/ --recursive
aws s3 ls s3://your-bucket/heavy/ --recursive

# Check results
aws s3 ls s3://your-bucket/results/ --recursive
```

### View Logs
```bash
# CloudWatch logs for jobs
aws logs tail /aws/batch/ocr-processing --follow

# Supervisor logs
aws s3 ls s3://your-bucket/supervisor_logs/ --recursive
```

## 🆘 Troubleshooting

### Common Issues

#### 1. Job Stuck in RUNNABLE State
**Problem**: Job shows RUNNABLE but never starts
**Solution**: Check compute environment capacity
```bash
aws batch describe-compute-environment --compute-environment ocr-compute-env
```

#### 2. Job Fails Immediately
**Problem**: Job goes to FAILED state right away
**Solution**: Check CloudWatch logs
```bash
aws logs tail /aws/batch/ocr-processing --follow
```

#### 3. Cannot Pull Docker Image
**Problem**: Job fails with image pull errors
**Solution**: Check ECR permissions and image exists
```bash
aws ecr list-images --repository-name ocr-pipeline
```

#### 4. S3 Access Denied
**Problem**: Job cannot read/write to S3
**Solution**: Check IAM role permissions
```bash
aws iam get-role-policy --role-name OCRBatchJobRole --policy-name OCRBatchJobPolicy
```

### Getting Help

1. **Check the logs first**: Always start with CloudWatch logs
2. **Verify configuration**: Double-check your `.env` file values
3. **Test locally**: Run jobs locally before deploying to AWS
4. **Check AWS limits**: Ensure you haven't hit service limits

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| AWS_REGION | AWS region | us-east-1 |
| S3_BUCKET | Your S3 bucket name | your-ocr-bucket |
| STANDARD_QUEUE | Standard processing queue | ocr-standard-queue |
| HIGH_MEMORY_QUEUE | High-memory queue | ocr-high-memory-queue |
| ESCALATION_CONFIDENCE | Min confidence threshold | 0.8 |
| ESCALATION_MAX_PAGES | Max pages before escalation | 50 |
| ESCALATION_MAX_TIME | Max processing time (seconds) | 600 |
| MAX_RETRIES | Maximum retry attempts | 3 |
| COST_OPTIMIZATION | Enable cost savings mode | false |

### Cost Optimization

Set `COST_OPTIMIZATION=true` to reduce resource usage:
- Smaller instance types
- Tighter escalation thresholds
- Fewer retry attempts

## 📈 Scaling and Performance

### Instance Types

**Standard Queue**: t3.medium, t3.large (cost-effective)
**High Memory Queue**: m5.large, m5.xlarge (for heavy processing)

### Scaling Based on Load

- **Light usage**: Keep default settings
- **Heavy usage**: Increase max vCPUs in compute environment
- **Cost conscious**: Use spot instances (modify compute environment)

### Monitoring Costs

```bash
# Check Batch costs
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-31 \
    --metrics BlendedCost \
    --group-by Type=DIMENSION,Key=SERVICE \
    --filter '{
        "Dimensions": {
            "Key": "SERVICE",
            "Values": ["AWS Batch", "Amazon Elastic Compute Cloud - Compute"]
        }
    }'
```

## 🔄 Updating Your Pipeline

When you update the OCR code:

1. **Test locally first**
2. **Build new Docker image**: `./scripts/deploy.sh`
3. **Update job definition** (automatic in deploy script)
4. **Test with small files**
5. **Monitor logs during rollout**

## 📞 Notifications Setup

### Discord Notifications

1. Create a Discord webhook URL
2. Replace `PLACEHOLDER_DISCORD_WEBHOOK` in your `.env` file
3. Supervisor will send status updates to your Discord channel

### Email Notifications

1. Set up SMTP server details
2. Replace placeholders in `.env` file
3. Emails will be sent for escalation events

## 🎯 Next Steps

1. **Set up supervisor Lambda**: Schedule automatic escalation checking
2. **Configure notifications**: Add Discord webhook and email settings
3. **Monitor regularly**: Check logs and S3 folders daily initially
4. **Optimize costs**: Adjust instance types based on usage patterns
5. **Add more automation**: Set up CloudWatch alarms and dashboards

---

**Need Help?** Start with the troubleshooting section above, then check AWS documentation for Batch/ECR/S3 services.

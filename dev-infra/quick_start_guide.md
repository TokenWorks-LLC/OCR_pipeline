# 🚀 OCR Pipeline AWS Deployment - Beginner's Guide

**Time to complete: 30-45 minutes**

This guide will walk you through deploying your OCR pipeline to AWS as a complete beginner. We'll use AWS CloudShell (recommended) or your local Windows setup.

## 🎯 What You'll Accomplish

By the end of this guide, you'll have:
- ✅ AWS infrastructure set up (S3, Batch, IAM)
- ✅ Docker container deployed to ECR
- ✅ OCR pipeline running in AWS Batch
- ✅ Automatic escalation system for failed jobs
- ✅ Monitoring and logging in place

## 📋 Prerequisites Check

### Required Software
- [x] **AWS Account** - You have access to AWS Console
- [ ] **AWS CLI** - We'll install/configure this
- [ ] **Docker Desktop** - We'll use this for containerization

### Required Knowledge
- Basic command line usage
- Understanding of files and folders
- No prior AWS experience needed!

---

## 🚀 Step-by-Step Deployment

### Step 1: Access AWS CloudShell (Easiest Method)

1. **Go to AWS Console**
   - Open https://console.aws.amazon.com/
   - Sign in to your AWS account

2. **Open CloudShell**
   - Click the terminal icon (>) in the bottom navigation bar
   - If you don't see it, search for "CloudShell" in the search bar
   - CloudShell gives you a Linux terminal in your browser!

3. **Clone Your Repository**
   ```bash
   git clone https://github.com/your-username/ocr-pipeline.git
   cd ocr-pipeline/dev-infra
   ```

   *Note: Replace `your-username` with your actual GitHub username*

### Step 2: Configure AWS (First Time Only)

1. **Set up AWS CLI** (in CloudShell):
   ```bash
   aws configure
   ```

2. **Enter your credentials**:
   - **AWS Access Key ID**: Get from AWS Console → IAM → Users → Your User → Security credentials
   - **AWS Secret Access Key**: Same location as above
   - **Default region name**: `us-east-1` (N. Virginia)
   - **Default output format**: `json`

   ```
   AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
   AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
   Default region name [None]: us-east-1
   Default output format [None]: json
   ```

### Step 3: Create AWS Infrastructure

**Run the automated setup script**:

```bash
chmod +x infrastructure/aws_batch_setup.sh
./infrastructure/aws_batch_setup.sh
```

This script will create:
- 🪣 **S3 Bucket** for file storage
- 🔐 **IAM Role** for Batch jobs
- 🚀 **Batch Queues** (standard + high-memory)
- 📊 **CloudWatch Logs** for monitoring

**Expected output:**
```
🚀 Setting up AWS infrastructure for OCR Pipeline
Region: us-east-1
Account: 123456789012
S3 Bucket: tokenworks-ocr-1694300000
✅ AWS infrastructure setup complete!

📋 Summary:
S3 Bucket: tokenworks-ocr-1694300000
Standard Queue: ocr-standard-queue
High Memory Queue: ocr-high-memory-queue
Job Definition: ocr-processing-job
```

**Important:** Save your S3 bucket name! You'll need it later.

### Step 4: Configure Environment Variables

1. **Copy the example file**:
   ```bash
   cp env.example .env
   ```

2. **Edit the .env file**:
   ```bash
   nano .env
   ```

3. **Update these key values**:
   ```bash
   # Replace with your actual bucket name from Step 3
   S3_BUCKET=tokenworks-ocr-1694300000

   # Keep these as-is for now (placeholders for founder approval)
   DISCORD_WEBHOOK_URL=PLACEHOLDER_DISCORD_WEBHOOK
   EMAIL_SMTP_SERVER=PLACEHOLDER_SMTP_SERVER
   EMAIL_RECIPIENTS=your-email@domain.com
   ```

4. **Save and exit** (Ctrl+X, then Y, then Enter in nano)

### Step 5: Build and Deploy Docker Container

**Run the deployment script**:

```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

This will:
- 🏗️ **Build** your Docker container with all OCR code
- 📤 **Push** it to Amazon ECR (Elastic Container Registry)
- 🔄 **Update** the Batch job definition

**Expected output:**
```
🚀 Deploying OCR Pipeline to AWS
🏗️ Building Docker image...
📤 Pushing to ECR...
✅ Deployment complete!
ECR Image: 123456789012.dkr.ecr.us-east-1.amazonaws.com/ocr-pipeline:latest
Ready to submit jobs!
```

---

## 🧪 Test Your Deployment

### Test 1: Upload a Sample File

```bash
# Upload the test image from your project
aws s3 cp ../data/samples/test.png s3://YOUR_BUCKET_NAME/input/
```

Replace `YOUR_BUCKET_NAME` with your actual bucket name.

### Test 2: Submit Your First Job

```bash
chmod +x scripts/submit_job.sh
./scripts/submit_job.sh s3://YOUR_BUCKET_NAME/input/test.png
```

**Expected output:**
```
✅ Job submitted: ocr-job-1694300000
```

### Test 3: Monitor Job Progress

```bash
# Check job status
aws batch list-jobs --job-queue ocr-standard-queue

# Check CloudWatch logs
aws logs tail /aws/batch/ocr-processing --follow
```

### Test 4: Check Results

```bash
# List result files
aws s3 ls s3://YOUR_BUCKET_NAME/results/ --recursive

# Download and view results
aws s3 cp s3://YOUR_BUCKET_NAME/results/YOUR_JOB_ID.json .
cat YOUR_JOB_ID.json
```

---

## 🔍 Understanding Your Pipeline

### How Jobs Flow Through the System

1. **📤 Job Submission**: You submit files via the script
2. **⚙️ Processing**: AWS Batch runs your container on EC2 instances
3. **🤖 OCR Pipeline**: Your existing OCR code processes the file
4. **📊 Results Check**: System evaluates if results meet quality thresholds:
   - ✅ **Good results** → Saved to `results/` folder
   - ❌ **Poor results** → Moved to `failed/` or `heavy/` folders

### Automatic Escalation System

Every 10 minutes, the supervisor checks for failed jobs and:
- 🔄 **Retries** failed jobs up to 3 times
- 💪 **Escalates** to high-memory instances for large/slow files
- 📋 **Flags** unfixable jobs for manual review

### S3 Folder Structure

```
your-bucket-name/
├── input/           # Files you upload
├── results/         # Successful processing results
├── failed/          # Jobs needing retry (low confidence)
├── heavy/           # Large/slow jobs (50+ pages, 10+ minutes)
├── processing/      # Jobs currently being retried
├── manual_review/   # Jobs requiring human intervention
└── supervisor_logs/ # Automatic system reports
```

---

## 📊 Monitoring Your Pipeline

### Real-time Monitoring Commands

```bash
# Check all jobs in your queues
aws batch describe-job-queues --job-queues ocr-standard-queue ocr-high-memory-queue

# See detailed job information
aws batch describe-jobs --jobs YOUR_JOB_ID

# Monitor logs in real-time
aws logs tail /aws/batch/ocr-processing --follow

# Check for escalated jobs
aws s3 ls s3://YOUR_BUCKET/failed/ --recursive
aws s3 ls s3://YOUR_BUCKET/heavy/ --recursive
```

### Daily Health Check

Run this every morning to check system health:

```bash
echo "=== DAILY HEALTH CHECK ==="
echo "Jobs in queues:"
aws batch list-jobs --job-queue ocr-standard-queue --max-results 5
aws batch list-jobs --job-queue ocr-high-memory-queue --max-results 5

echo -e "\nEscalated jobs:"
aws s3 ls s3://YOUR_BUCKET/failed/ --recursive | wc -l
aws s3 ls s3://YOUR_BUCKET/heavy/ --recursive | wc -l

echo -e "\nCompleted jobs today:"
aws s3 ls s3://YOUR_BUCKET/results/ --recursive | grep $(date +%Y-%m-%d) | wc -l
```

---

## 🆘 Troubleshooting Common Issues

### Issue: Job Stuck in "RUNNABLE" State

**Problem**: Job shows RUNNABLE but never starts processing.

**Solution**: Check if compute environment has capacity.
```bash
aws batch describe-compute-environment --compute-environment ocr-compute-env
```
**Fix**: The environment might need more instances. Wait a few minutes or check AWS limits.

### Issue: Job Fails Immediately

**Problem**: Job goes to FAILED state right after submission.

**Solution**: Check the logs.
```bash
aws logs tail /aws/batch/ocr-processing --follow
```
**Common fixes**:
- Check if Docker image exists in ECR
- Verify S3 bucket permissions
- Ensure environment variables are set correctly

### Issue: Cannot Access S3

**Problem**: Job cannot read/write files.

**Solution**: Check IAM permissions.
```bash
aws iam get-role-policy --role-name OCRBatchJobRole --policy-name OCRBatchJobPolicy
```
**Fix**: The IAM role should have S3 permissions. If not, re-run the setup script.

### Issue: Docker Build Fails

**Problem**: `deploy.sh` fails during Docker build.

**Solution**: Check Docker is running and you have sufficient disk space.
```bash
docker system df
docker system prune -f  # Clean up if needed
```

---

## 🎯 Advanced Setup (Optional)

### Setting Up Supervisor Automation

The supervisor automatically handles failed jobs, but you need to set it up:

1. **Create Lambda Function** in AWS Console
2. **Upload** `infrastructure/supervisor_lambda.py`
3. **Add CloudWatch Event Rule** to trigger every 10 minutes

### Adding Notifications

1. **Discord Webhook**:
   - Create webhook in your Discord server
   - Update `DISCORD_WEBHOOK_URL` in `.env`

2. **Email Notifications**:
   - Set up SMTP server details
   - Update email settings in `.env`

---

## 💰 Cost Monitoring

### Check Your AWS Costs

```bash
# Monthly costs for Batch and EC2
aws ce get-cost-and-usage \
  --time-period Start=2024-10-01,End=2024-10-31 \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE \
  --filter '{"Dimensions":{"Key":"SERVICE","Values":["AWS Batch","Amazon Elastic Compute Cloud - Compute"]}}'
```

### Cost Optimization Tips

- **Set `COST_OPTIMIZATION=true`** in `.env` for smaller instances
- **Use spot instances** for non-critical workloads
- **Monitor usage** and adjust instance types based on needs

---

## 📞 Getting Help

1. **Check this guide first** - Most issues are covered above
2. **Check AWS documentation** for Batch/ECR/S3 services
3. **Review CloudWatch logs** - They contain detailed error messages
4. **Test locally first** - Run your pipeline locally before deploying

---

## ✅ Success Checklist

After completing this guide, verify you have:

- [ ] AWS infrastructure created (S3, Batch, IAM)
- [ ] Docker container deployed to ECR
- [ ] Successfully processed at least one test file
- [ ] Can monitor jobs and view results
- [ ] Understand the escalation system
- [ ] Know how to troubleshoot common issues

**Congratulations! 🎉** You now have a production-ready OCR pipeline running on AWS with intelligent failure handling and automatic scaling.

---

*Need help? Start with the troubleshooting section above, then check the detailed README.md for more advanced topics.*

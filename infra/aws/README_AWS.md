# AWS Batch Deployment Guide for Page-Text Pipeline

This guide describes how to deploy and run the page-text extraction pipeline on AWS Batch for parallel processing of the entire corpus (189,354 pages).

## 📋 Overview

**Architecture**: Single AWS Batch queue processing N manifest shards in parallel
**No Escalation**: Uniform processing (no high-memory queue or supervisor Lambda)
**Output**: Individual CSV files per shard → aggregated into single client CSV

## 🔧 Prerequisites

1. **AWS Account** with permissions for:
   - ECR (Elastic Container Registry)
   - Batch (job queues, compute environments, job definitions)
   - S3 (bucket read/write)
   - IAM (service roles)

2. **Local Tools**:
   - AWS CLI v2 configured (`aws configure`)
   - Docker Desktop running
   - Git (for repo access)
   - `jq` (JSON processor for bash scripts)

3. **AWS Resources** (must be created beforehand or use placeholders):
   - VPC with subnets (`<SUBNET_IDS_COMMA>`)
   - Security group (`<SG_ID_COMMA>`)
   - EC2 key pair (`<KEYPAIR_NAME>`)
   - S3 buckets for input/output

## 🚀 Deployment Steps

### Step 1: Configure Environment Variables

Copy the example file and fill in your values:

```bash
cp infra/aws/vars.env.example infra/aws/vars.env
```

Edit `infra/aws/vars.env` with real values:

```bash
AWS_REGION=us-east-1
ACCOUNT_ID=123456789012                              # Your AWS account ID
ECR_REPO=ocr-page-text
IMAGE_TAG=pt-20251009_1951                           # Unique tag per deployment
S3_INPUT=s3://your-bucket/secondary-sources/         # Where manifest shards live
S3_OUTPUT=s3://your-bucket/page-text-runs/           # Where results go
SHARDS=100                                           # Number of parallel jobs
VCPU=2                                               # vCPUs per job
MEMORY_MB=4096                                       # Memory per job (4GB)
JOB_QUEUE_NAME=page-text-queue
JOB_DEF_NAME=page-text-jobdef
```

### Step 2: Build and Push Docker Image

Load environment and build:

```bash
# Load variables
set -a; source infra/aws/vars.env; set +a

# Build and push to ECR
bash infra/aws/ecr_build_push.sh
```

**Expected Output**:
```
IMAGE_URI=123456789012.dkr.ecr.us-east-1.amazonaws.com/ocr-page-text:pt-20251009_1951
```

### Step 3: Provision AWS Batch Infrastructure

**⚠️ IMPORTANT**: Before running this script, you must:
1. Have a VPC with subnets and security group created
2. Update `infra/aws/batch_simple.sh` placeholders:
   - `<SUBNET_IDS_COMMA>` → `subnet-abc123,subnet-def456`
   - `<SG_ID_COMMA>` → `sg-xyz789`
   - `<KEYPAIR_NAME>` → `my-keypair`

Run the provisioning script:

```bash
bash infra/aws/batch_simple.sh
```

**Expected Output**:
```
OK: queue=page-text-queue, jobdef=page-text-jobdef, image=123456789012.dkr.ecr.us-east-1.amazonaws.com/ocr-page-text:pt-20251009_1951
```

### Step 4: Prepare and Upload Manifest Shards

**Create Shards**:
```bash
# Shard the full manifest (189,354 pages → 100 shards of ~1,894 pages each)
python tools/shard_manifest.py secondary_sources_full_20251009_0949.txt 100

# Verify shards created
ls -lh manifests/shard_*.txt | wc -l   # Should show 100
```

**Upload to S3**:
```bash
# Upload manifest shards
aws s3 sync manifests/ "$S3_INPUT/manifests/" --exclude "*" --include "shard_*.txt"

# Verify upload
aws s3 ls "$S3_INPUT/manifests/" | wc -l   # Should show 100 files
```

### Step 5: Submit Batch Jobs

Submit all 100 jobs (one per shard):

```bash
bash infra/aws/submit_shards.sh
```

**Expected Output**:
```
submitted 100 shards
```

### Step 6: Monitor Job Progress

**Check Job Status**:
```bash
# List running jobs
aws batch list-jobs --job-queue page-text-queue --job-status RUNNING

# List succeeded jobs
aws batch list-jobs --job-queue page-text-queue --job-status SUCCEEDED

# List failed jobs (should be zero)
aws batch list-jobs --job-queue page-text-queue --job-status FAILED
```

**CloudWatch Logs**:
```bash
# Tail logs for all jobs
aws logs tail /aws/batch/page-text-processing --follow

# Check specific job logs
aws batch describe-jobs --jobs <job-id> | jq -r '.jobs[0].container.logStreamName'
aws logs tail /aws/batch/page-text-processing --log-stream-name <log-stream>
```

**Estimated Time**: 
- **Pages per shard**: 189,354 / 100 = ~1,894 pages
- **Median processing speed**: ~3 seconds/page (based on local tests)
- **Per-shard ETA**: 1,894 × 3 ÷ 60 = ~95 minutes (worst case)
- **With parallelization**: ~95 minutes total (all jobs run simultaneously)

### Step 7: Verify Outputs

**Quick Sanity Check** (after first 5 jobs complete):

```bash
# List completed shard outputs
aws s3 ls "$S3_OUTPUT" --recursive | grep client_page_text.csv | head -5

# Download one shard for spot-check
aws s3 cp "$S3_OUTPUT/page_text_pt-20251009_1951_shard_00/client_page_text.csv" test_shard_00.csv

# Verify has_akkadian detection on known-positive PDFs
grep -i "akkadian" test_shard_00.csv | head -5
```

**Expected**: Each shard should have:
- `client_page_text.csv` (4 columns: pdf_name, page, page_text, has_akkadian)
- `progress.csv` (tracking file)
- Rows corresponding to manifest shard size (~1,894 pages per shard)

### Step 8: Aggregate Results

**After All Jobs Complete**:

```bash
# Download all shard CSVs
aws s3 sync "$S3_OUTPUT" ./_downloads/ --exclude "*" --include "*/client_page_text.csv"

# Combine into single CSV
python tools/aggregate_page_text_csv.py "_downloads/**/client_page_text.csv" combined_page_text.csv

# Verify output
wc -l combined_page_text.csv   # Should be 189,355 (189,354 pages + 1 header)

# Upload to final location
aws s3 cp combined_page_text.csv s3://sek-ocr-reports/page_text_20251009_1951/client_page_text.csv
```

## 🔍 Troubleshooting

### Job Stuck in RUNNABLE State

**Problem**: Jobs show `RUNNABLE` but never start  
**Solution**: Check compute environment capacity

```bash
aws batch describe-compute-environments --compute-environment page-text-queue-ce
# Look for "state": "ENABLED", "status": "VALID"
# Check maxvCpus is sufficient (should be 256)
```

### Job Fails Immediately

**Problem**: Jobs go to `FAILED` state right away  
**Solution**: Check CloudWatch logs for specific error

```bash
aws batch describe-jobs --jobs <job-id>
# Copy logStreamName from output
aws logs tail /aws/batch/page-text-processing --log-stream-name <stream-name>
```

Common causes:
- Missing S3 manifest shard → Upload manifests first
- IAM role lacks S3 permissions → Add `s3:GetObject`, `s3:PutObject`
- Docker image not found → Re-run `ecr_build_push.sh`

### Shard Missing Output CSV

**Problem**: Job succeeded but no `client_page_text.csv` in S3  
**Solution**: Pull CloudWatch logs to see if worker failed silently

```bash
# Find job ID for shard
aws batch list-jobs --job-queue page-text-queue | jq -r '.jobSummaryList[] | select(.jobName | contains("shard_05"))'

# Get logs
aws batch describe-jobs --jobs <job-id> | jq -r '.jobs[0].container.logStreamName'
aws logs get-log-events --log-group-name /aws/batch/page-text-processing --log-stream-name <stream>
```

Check for:
- PDF files not accessible (fix manifest paths)
- Out of memory (increase `MEMORY_MB` in vars.env)
- Python exceptions (fix code or skip problematic PDFs)

### S3 Access Denied

**Problem**: Worker cannot read manifest or write results  
**Solution**: Verify IAM job role permissions

```bash
aws iam get-role --role-name OCRBatchJobRole
aws iam list-attached-role-policies --role-name OCRBatchJobRole
```

Required permissions:
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject",
    "s3:PutObject",
    "s3:ListBucket"
  ],
  "Resource": [
    "arn:aws:s3:::your-bucket/*",
    "arn:aws:s3:::your-bucket"
  ]
}
```

## 💰 Cost Optimization

### Reduce Costs

1. **Use Spot Instances**: Edit `batch_simple.sh` to use `type=SPOT` instead of `ON_DEMAND`
2. **Reduce vCPUs**: Lower `VCPU=1` and `MEMORY_MB=2048` for smaller jobs
3. **Fewer Shards**: Use `SHARDS=50` (longer per-job runtime but fewer jobs)
4. **Terminate Compute Environment**: After run completes:
   ```bash
   aws batch update-compute-environment --compute-environment page-text-queue-ce --state DISABLED
   ```

### Monitor Costs

```bash
# Check AWS Batch costs for current month
aws ce get-cost-and-usage \
    --time-period Start=2025-10-01,End=2025-10-31 \
    --metrics BlendedCost \
    --granularity MONTHLY \
    --group-by Type=DIMENSION,Key=SERVICE \
    --filter '{
        "Dimensions": {
            "Key": "SERVICE",
            "Values": ["AWS Batch", "Amazon Elastic Container Registry"]
        }
    }'
```

## 📊 Quality Gates

Before delivering final CSV, verify:

1. **Row Count**: `wc -l combined_page_text.csv` should be 189,355 (189,354 + header)
2. **Akkadian Detection**: Spot-check known-positive PDFs have `has_akkadian=1`
3. **No Truncation**: Random sample pages should have full text (not truncated mid-sentence)
4. **UTF-8 BOM**: File should start with `0xEF 0xBB 0xBF` (UTF-8 BOM marker)

```bash
# Check BOM
xxd combined_page_text.csv | head -1
# Should show: ef bb bf (first 3 bytes)

# Check row count
wc -l combined_page_text.csv

# Spot-check Akkadian detection (adjust grep pattern for known PDF names)
grep -E "(cuneiform|akkadian|babylonian)" combined_page_text.csv --ignore-case | head -10
```

## 🎯 Next Steps

1. **Scale Up**: For faster runs, increase `SHARDS=200` (more parallelization)
2. **Automate**: Create Lambda trigger to submit jobs on S3 manifest upload
3. **Dashboards**: Build CloudWatch dashboard for job monitoring
4. **Alerts**: Set up SNS alerts for job failures
5. **Reusable**: Save `vars.env` as template for future corpus runs

---

**Support**: Check `cloud/worker/page_text_worker.py` for worker logic, `tools/run_page_text.py` for extraction pipeline.

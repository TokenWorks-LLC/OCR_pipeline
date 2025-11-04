# Page-Text Pipeline AWS Adaptation Guide

**Branch**: `integrate/aws_scaffold_20251009_1943`  
**Date**: 2024-10-09  
**Status**: Planning Phase

---

## 📋 Executive Summary

This document describes how to adapt the teammate's AWS Batch infrastructure (`dev-infra/`) for the **page-text CSV generation pipeline** (`tools/run_page_text.py`). The goal is to generate a single 4-column CSV file across the entire corpus (189,354 pages) quickly using AWS parallelization.

### Original Infrastructure Purpose
- **Comprehensive OCR pipeline**: Full document processing with OCR, LLM correction, Akkadian extraction
- **Escalation logic**: Moves failed jobs to high-memory queue based on confidence/size/time thresholds
- **Supervisor pattern**: Lambda function periodically reprocesses escalated jobs
- **S3 folder structure**: `input/`, `results/`, `failed/`, `heavy/`, `manual_review/`, `supervisor_logs/`

### Page-Text Pipeline Requirements
- **Simple text extraction**: PDF → page text with Akkadian detection (no LLM correction, no OCR images)
- **No escalation needed**: All pages can be processed uniformly (no heavy/light distinction)
- **Manifest-based sharding**: Split 189,354 pages across N parallel jobs
- **Single CSV output**: Aggregate all job results into one UTF-8 BOM CSV with 4 columns
- **S3 output**: `s3://sek-ocr-reports/page_text_<timestamp>/client_page_text.csv`

---

## 🔍 Infrastructure Analysis

### What We're Keeping

✅ **AWS Batch Setup** (`infrastructure/aws_batch_setup.sh`)
- Creates standard queue, compute environment, IAM roles
- Will REMOVE high-memory queue (not needed for page-text)
- Will REMOVE supervisor Lambda (no escalation logic)

✅ **Docker Deployment** (`scripts/deploy.sh`)
- ECR repository creation
- Docker build and push workflow
- Job definition registration
- Will MODIFY job definition to run `tools/run_page_text.py`

✅ **Environment Variables** (`env.example`)
- AWS region, S3 bucket configuration
- Will REMOVE escalation thresholds, notification placeholders
- Will ADD manifest sharding parameters

✅ **Dockerfile**
- Python 3.9 slim base (page-text uses 3.13 locally, but 3.9+ is compatible)
- System dependencies for OCR
- Will ADD PyMuPDF (fitz) and Pillow dependencies
- Will REMOVE Tesseract (not needed for page-text)
- Will CHANGE entrypoint to run `tools/run_page_text.py`

### What We're Removing

❌ **Escalation Logic**
- `cloud/supervisor/escalation_manager.py` - Not needed
- `cloud/supervisor/supervisor_runner.py` - Not needed
- `infrastructure/supervisor_lambda.py` - Not needed
- High-memory queue configuration

❌ **Notification System**
- Discord webhook placeholders
- Email SMTP configuration
- Job failure notifications (no failures expected with uniform processing)

❌ **S3 Folder Structure**
- `failed/`, `heavy/`, `processing/`, `manual_review/`, `supervisor_logs/` - Not needed
- Keep: `input/` (manifest shards), `results/` (partial CSVs), final output location

### What We're Adding

✨ **Manifest Sharding Tool** (`tools/shard_manifest.py`)
- Splits `secondary_sources_full_20251009_0949.txt` (189,354 pages) into N shards
- Each shard: `manifest_shard_001.txt`, `manifest_shard_002.txt`, ...
- Upload shards to `s3://SEK_OCR_BUCKET/page_text_input/<timestamp>/manifest_shard_*.txt`

✨ **Page-Text Worker** (`cloud/worker/page_text_worker.py`)
- Downloads manifest shard from S3
- Runs `tools/run_page_text.py` with `--manifest` flag
- Streams CSV output to S3: `s3://SEK_OCR_BUCKET/page_text_results/<timestamp>/shard_<N>.csv`
- No escalation logic, no retries (fail fast)

✨ **CSV Aggregation Script** (`tools/aggregate_page_text_csv.py`)
- Downloads all `shard_*.csv` files from S3
- Combines into single `client_page_text.csv` with UTF-8 BOM
- Uploads to `s3://sek-ocr-reports/page_text_<timestamp>/client_page_text.csv`
- Run locally after all jobs complete (not on AWS)

---

## 🛠️ Implementation Plan

### Phase 1: Modify Dockerfile ✅ (Next)

**File**: `dev-infra/Dockerfile`

**Changes**:
```dockerfile
# Change base image to Python 3.11 (closer to local 3.13, but AWS Lambda compatible)
FROM python:3.11-slim

# Remove Tesseract (not needed for page-text)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy page-text tool instead of production/
COPY tools/run_page_text.py /app/tools/run_page_text.py
COPY profiles/akkadian_strict.json /app/profiles/akkadian_strict.json

# Copy cloud worker
COPY dev-infra/cloud/worker/page_text_worker.py /app/cloud/worker/page_text_worker.py

# Install Python dependencies (add PyMuPDF, Pillow, boto3)
WORKDIR /app
RUN pip install --no-cache-dir PyMuPDF Pillow numpy tqdm boto3 requests

# Entrypoint: page-text worker
ENTRYPOINT ["python", "-m", "cloud.worker.page_text_worker"]
```

### Phase 2: Create Page-Text Worker ✅

**File**: `dev-infra/cloud/worker/page_text_worker.py`

**Purpose**: Download manifest shard, run page-text pipeline, upload CSV

**Logic**:
1. Parse job parameters: `{"manifest_shard_s3": "s3://bucket/input/manifest_shard_001.txt", "shard_id": "001"}`
2. Download manifest shard from S3 to `/tmp/manifest_shard_001.txt`
3. Run `tools/run_page_text.py --manifest /tmp/manifest_shard_001.txt --output /tmp/shard_001.csv --profile profiles/akkadian_strict.json --no-progress`
4. Upload `/tmp/shard_001.csv` to `s3://SEK_OCR_BUCKET/page_text_results/<timestamp>/shard_001.csv`
5. Exit with code 0 (success) or non-zero (failure)

### Phase 3: Create Manifest Sharding Tool ✅

**File**: `tools/shard_manifest.py`

**Purpose**: Split large manifest into N equal shards

**Usage**:
```bash
python tools/shard_manifest.py \
    --manifest secondary_sources_full_20251009_0949.txt \
    --num-shards 100 \
    --output-dir manifest_shards/
```

**Output**: 
- `manifest_shards/manifest_shard_001.txt` (1,894 pages)
- `manifest_shards/manifest_shard_002.txt` (1,894 pages)
- ... 
- `manifest_shards/manifest_shard_100.txt` (1,894 pages)

### Phase 4: Simplify env.example ✅

**File**: `dev-infra/env.example`

**New Content**:
```bash
# AWS Configuration
AWS_REGION=us-east-1
S3_BUCKET=PLACEHOLDER_SEK_OCR_BUCKET

# Page-Text Job Configuration
NUM_SHARDS=100
MANIFEST_NAME=secondary_sources_full_20251009_0949.txt

# Output Configuration
OUTPUT_BUCKET=sek-ocr-reports
OUTPUT_PREFIX=page_text

# Job Queue (no high-memory queue needed)
STANDARD_QUEUE=page-text-standard-queue
```

### Phase 5: Simplify aws_batch_setup.sh ✅

**File**: `dev-infra/infrastructure/aws_batch_setup.sh`

**Changes**:
- Remove high-memory queue creation
- Remove supervisor Lambda creation
- Rename queue to `page-text-standard-queue`
- Update IAM role to only allow S3 read/write (no Batch escalation permissions)

### Phase 6: Modify deploy.sh ✅

**File**: `dev-infra/scripts/deploy.sh`

**Changes**:
- Update job definition name: `page-text-job` (not `ocr-processing-job`)
- Update ECR repository: `page-text-pipeline` (not `ocr-pipeline`)
- Update container properties: 2 vCPUs, 4 GB memory (sufficient for page-text)

### Phase 7: Create Batch Submission Script ✅

**File**: `dev-infra/scripts/submit_page_text_jobs.sh`

**Purpose**: Upload manifest shards and submit N jobs to AWS Batch

**Usage**:
```bash
# Upload manifest shards to S3
aws s3 sync manifest_shards/ s3://$S3_BUCKET/page_text_input/20241009_2100/

# Submit 100 jobs (one per shard)
for i in {001..100}; do
    aws batch submit-job \
        --job-name "page-text-shard-$i" \
        --job-queue page-text-standard-queue \
        --job-definition page-text-job \
        --parameters manifest_shard_s3="s3://$S3_BUCKET/page_text_input/20241009_2100/manifest_shard_$i.txt",shard_id="$i"
done
```

### Phase 8: Create CSV Aggregation Tool ✅

**File**: `tools/aggregate_page_text_csv.py`

**Purpose**: Combine all shard CSVs into single client CSV

**Usage** (run locally after all jobs complete):
```bash
python tools/aggregate_page_text_csv.py \
    --s3-prefix s3://SEK_OCR_BUCKET/page_text_results/20241009_2100/ \
    --output-s3 s3://sek-ocr-reports/page_text_20241009_2100/client_page_text.csv \
    --local-output client_page_text.csv
```

**Logic**:
1. List all files in S3 prefix: `s3://SEK_OCR_BUCKET/page_text_results/20241009_2100/shard_*.csv`
2. Download each shard CSV to `/tmp/`
3. Read all CSVs, sort by (pdf_name, page) for deterministic ordering
4. Write single CSV with UTF-8 BOM header
5. Upload to `s3://sek-ocr-reports/page_text_20241009_2100/client_page_text.csv`
6. Save local copy to `client_page_text.csv`

---

## 📊 Execution Workflow

### Local Setup (One-Time)

1. **Install AWS CLI**:
   ```powershell
   # Windows: Download installer from https://aws.amazon.com/cli/
   aws configure
   # Enter: Access Key, Secret Key, us-east-1, json
   ```

2. **Create manifest shards**:
   ```powershell
   python tools/shard_manifest.py `
       --manifest secondary_sources_full_20251009_0949.txt `
       --num-shards 100 `
       --output-dir manifest_shards/
   ```

3. **Set up AWS infrastructure**:
   ```bash
   cd dev-infra
   chmod +x infrastructure/aws_batch_setup.sh scripts/deploy.sh
   ./infrastructure/aws_batch_setup.sh
   ```

4. **Build and deploy Docker image**:
   ```bash
   ./scripts/deploy.sh
   ```

### AWS Execution (Per Run)

1. **Upload manifest shards to S3**:
   ```powershell
   $timestamp = Get-Date -Format "yyyyMMdd_HHmm"
   aws s3 sync manifest_shards/ s3://PLACEHOLDER_SEK_OCR_BUCKET/page_text_input/$timestamp/
   ```

2. **Submit batch jobs**:
   ```bash
   # Use submit_page_text_jobs.sh with timestamp from step 1
   ./scripts/submit_page_text_jobs.sh $timestamp
   ```

3. **Monitor job progress**:
   ```bash
   # Check queue status
   aws batch list-jobs --job-queue page-text-standard-queue --job-status RUNNING

   # Check CloudWatch logs
   aws logs tail /aws/batch/page-text-processing --follow
   ```

4. **Wait for all jobs to complete** (estimated: 10-30 minutes for 189K pages)

5. **Aggregate results locally**:
   ```powershell
   python tools/aggregate_page_text_csv.py `
       --s3-prefix s3://PLACEHOLDER_SEK_OCR_BUCKET/page_text_results/$timestamp/ `
       --output-s3 s3://sek-ocr-reports/page_text_$timestamp/client_page_text.csv `
       --local-output client_page_text.csv
   ```

---

## 🚦 Testing Strategy

### Phase 1: Local Docker Build
```powershell
# Build Docker image locally
docker build -f dev-infra/Dockerfile -t page-text-pipeline .

# Test with 3-page manifest shard
docker run --rm `
    -v ${PWD}/manifest_shards:/app/manifest_shards `
    -v ${PWD}/data:/app/data `
    page-text-pipeline `
    --manifest /app/manifest_shards/manifest_shard_001.txt `
    --output /tmp/shard_001.csv `
    --profile /app/profiles/akkadian_strict.json
```

### Phase 2: AWS Batch with Single Shard
```bash
# Submit one job to AWS Batch
aws batch submit-job \
    --job-name "page-text-test-shard-001" \
    --job-queue page-text-standard-queue \
    --job-definition page-text-job \
    --parameters manifest_shard_s3="s3://$S3_BUCKET/page_text_input/test/manifest_shard_001.txt",shard_id="001"

# Monitor logs
aws logs tail /aws/batch/page-text-processing --follow
```

### Phase 3: Full Production Run (100 Shards)
```bash
# Submit all 100 jobs
./scripts/submit_page_text_jobs.sh 20241009_2100
```

---

## 🔒 Placeholder Strategy

All placeholders will be replaced during deployment (not committed to Git):

| Placeholder | Value | Location |
|-------------|-------|----------|
| `PLACEHOLDER_SEK_OCR_BUCKET` | Founder provides | `dev-infra/.env` |
| `PLACEHOLDER_AWS_REGION` | `us-east-1` (default) | `dev-infra/.env` |
| `PLACEHOLDER_NUM_SHARDS` | `100` (default) | `dev-infra/.env` |

**Deployment Script** (`dev-infra/scripts/replace_placeholders.sh`):
```bash
#!/bin/bash
# Replace placeholders in .env file (run before deployment)

read -p "Enter SEK OCR S3 Bucket Name: " SEK_BUCKET
sed -i "s/PLACEHOLDER_SEK_OCR_BUCKET/$SEK_BUCKET/g" dev-infra/.env

echo "✅ Placeholders replaced. Ready to deploy."
```

---

## 📁 File Checklist

### New Files to Create
- [ ] `dev-infra/cloud/worker/page_text_worker.py` - AWS Batch worker for page-text pipeline
- [ ] `tools/shard_manifest.py` - Manifest splitting tool
- [ ] `tools/aggregate_page_text_csv.py` - CSV aggregation tool
- [ ] `dev-infra/scripts/submit_page_text_jobs.sh` - Batch job submission script
- [ ] `dev-infra/scripts/replace_placeholders.sh` - Placeholder replacement helper
- [ ] `dev-infra/PAGE_TEXT_QUICKSTART.md` - Simplified deployment guide

### Files to Modify
- [ ] `dev-infra/Dockerfile` - Change entrypoint, dependencies, remove Tesseract
- [ ] `dev-infra/env.example` - Simplify to page-text parameters
- [ ] `dev-infra/infrastructure/aws_batch_setup.sh` - Remove high-memory queue, supervisor Lambda
- [ ] `dev-infra/scripts/deploy.sh` - Update job definition name, ECR repository
- [ ] `dev-infra/README.md` - Add page-text adaptation notes

### Files to Keep Unchanged
- ✅ `tools/run_page_text.py` (protected)
- ✅ `README.md` (protected)
- ✅ `README_docker.md` (protected)
- ✅ `OCR _PIPELINE_RUNBOOK.md` (protected)
- ✅ `profiles/akkadian_strict.json` (protected)
- ✅ `tests/test_akkadian_protection.py` (protected)
- ✅ `run_pipeline_simple.py` (protected)

---

## 📝 Next Steps

1. **Create `tools/shard_manifest.py`** - Implement manifest splitting logic
2. **Create `dev-infra/cloud/worker/page_text_worker.py`** - Implement AWS worker
3. **Modify `dev-infra/Dockerfile`** - Adapt for page-text dependencies
4. **Simplify `dev-infra/env.example`** - Remove escalation config
5. **Modify `dev-infra/infrastructure/aws_batch_setup.sh`** - Remove supervisor
6. **Modify `dev-infra/scripts/deploy.sh`** - Update job definition
7. **Create `dev-infra/scripts/submit_page_text_jobs.sh`** - Batch submission
8. **Create `tools/aggregate_page_text_csv.py`** - CSV aggregation
9. **Test local Docker build** - Verify `docker build` works
10. **Commit changes** - `chore(aws): adapt infrastructure for page-text pipeline`
11. **Open PR** - Create PR with deployment guide and run instructions

---

**Author**: GitHub Copilot  
**Date**: 2024-10-09  
**Branch**: `integrate/aws_scaffold_20251009_1943`

#!/bin/bash
# Submit OCR job to AWS Batch

if [ $# -lt 1 ]; then
    echo "Usage: $0 <s3-input-path> [job-name]"
    echo "Example: $0 s3://my-bucket/input/document.pdf"
    exit 1
fi

S3_INPUT_PATH=$1
JOB_NAME=${2:-"ocr-job-$(date +%s)"}

# Submit job to AWS Batch
aws batch submit-job \
    --job-name "$JOB_NAME" \
    --job-queue ocr-standard-queue \
    --job-definition ocr-processing-job \
    --parameters "{\"job_data\": \"{\\\\"s3_input_path\\\": \\\"$S3_INPUT_PATH\\\", \\\"job_id\\\": \\\"$JOB_NAME\\\"}\"}"

echo "✅ Job submitted: $JOB_NAME"

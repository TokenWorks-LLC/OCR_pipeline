#!/usr/bin/env bash
set -euo pipefail
: "${AWS_REGION:?}" "${JOB_QUEUE_NAME:?}" "${JOB_DEF_NAME:?}" "${SHARDS:?}" \
  "${S3_INPUT:?}" "${S3_OUTPUT:?}" "${IMAGE_TAG:?}"
for i in $(seq -f "%02g" 0 $((SHARDS-1))); do
  SHARD_KEY="manifests/shard_${i}.txt"
  aws batch submit-job --region "$AWS_REGION" \
    --job-name "pt-${IMAGE_TAG}-${i}" \
    --job-queue "$JOB_QUEUE_NAME" \
    --job-definition "$JOB_DEF_NAME" \
    --container-overrides "environment=[{name=S3_INPUT,value=${S3_INPUT}},{name=S3_OUTPUT,value=${S3_OUTPUT}},{name=SHARD_KEY,value=${SHARD_KEY}},{name=SHARD_IDX,value=${i}},{name=IMAGE_TAG,value=${IMAGE_TAG}}]"
done
echo "submitted $SHARDS shards"

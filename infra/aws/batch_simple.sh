#!/usr/bin/env bash
set -euo pipefail
: "${AWS_REGION:?}" "${ACCOUNT_ID:?}" "${ECR_REPO:?}" "${IMAGE_TAG:?}" \
  "${JOB_QUEUE_NAME:?}" "${JOB_DEF_NAME:?}" "${VCPU:?}" "${MEMORY_MB:?}"

IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

# Create compute env if missing (EC2 managed, on-demand; minimal placeholders)
CE_NAME="${JOB_QUEUE_NAME}-ce"
aws batch describe-compute-environments --compute-environments "$CE_NAME" --region "$AWS_REGION" \
  | jq -e '.computeEnvironments[]?' >/dev/null 2>&1 || \
aws batch create-compute-environment --region "$AWS_REGION" \
  --compute-environment-name "$CE_NAME" --type MANAGED \
  --compute-resources type=ON_DEMAND,minvCpus=0,maxvCpus=256,desiredvCpus=0,instanceTypes=m5.large,subnets=<SUBNET_IDS_COMMA>,securityGroupIds=<SG_ID_COMMA>,allocationStrategy=BEST_FIT_PROGRESSIVE,ec2KeyPair=<KEYPAIR_NAME> \
  --service-role AWSBatchServiceRole >/dev/null

# Create job queue
aws batch describe-job-queues --job-queues "$JOB_QUEUE_NAME" --region "$AWS_REGION" \
  | jq -e '.jobQueues[]?' >/dev/null 2>&1 || \
aws batch create-job-queue --job-queue-name "$JOB_QUEUE_NAME" --state ENABLED \
  --priority 1 --compute-environment-order order=1,computeEnvironment="$CE_NAME" --region "$AWS_REGION" >/dev/null

# Register job definition
aws batch register-job-definition --job-definition-name "$JOB_DEF_NAME" --type container \
  --container-properties "image=${IMAGE_URI},vcpus=${VCPU},memory=${MEMORY_MB},command=[\"bash\",\"-lc\",\"python cloud/worker/page_text_worker.py\"],environment=[{name=PYTHONUNBUFFERED,value=1}]" \
  --region "$AWS_REGION" >/dev/null

echo "OK: queue=${JOB_QUEUE_NAME}, jobdef=${JOB_DEF_NAME}, image=${IMAGE_URI}"

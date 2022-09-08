REGION="europe-west4"

#machine spec
MACHINE_TYPE="n1-standard-4"
REPLICA_COUNT=1
GPU_CARD="NVIDIA_TESLA_T4"
GPU_COUNT=1

NAME="omega_conf_test"

DISPLAY_NAME="${NAME}_$(date +'%Y_%m_%dT%H_%M')"

if [[ -z "${CONTAINER_IMAGE_URI}" ]]; then
  echo "Container image uri env set, exiting"
  exit(0)
else
fi

if [[ -z "${CONFIG_URL}" ]]; then
  echo "Config url env not set, exiting"
  exit(0)
else
fi

gsu

echo "staring training job with args"
echo "CONTAINER_IMAGE_URI: $CONTAINER_IMAGE_URI"
echo "DISPLAY_NAME: $DISPLAY_NAME"
# https://cloud.google.com/sdk/gcloud/reference/ai/custom-jobs/create

gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=${DISPLAY_NAME} \
  --worker-pool-spec=machine-type=${MACHINE_TYPE},replica-count=${REPLICA_COUNT},container-image-uri=${CONTAINER_IMAGE_URI},accelerator-type=${GPU_CARD},accelerator-count=${GPU_COUNT} \
  --args=--config-file-path=${CONFIG_URL}

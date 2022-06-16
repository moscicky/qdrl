# TODO: uncomment. Use autopackaging if pushing to eu gcr is possible
#EXECUTOR_IMAGE_URI="europe-docker.pkg.dev/vertex-ai/training/pytorch-xla.1-11:latest"
#
#if [[ -z "${PROJECT}" ]]; then
#  echo "Project env variable not set, exiting"
#  exit(0)
#else
#  CONTAINER_URI="eu.gcr.io/${PROJECT}/${DISPLAY_NAME}"
#fi
#echo "CONTAINER_URI: $CONTAINER_URI"

REGION="europe-west4"

#package spec
PYTHON_MODULE="qdrl.main_train"
WORKING_DIRECTORY="."

#machine spec
MACHINE_TYPE="n1-standard-4"
REPLICA_COUNT=1
GPU_CARD="NVIDIA_TESLA_P4"
GPU_COUNT=1

#job arguments
TASK_ID="chunked_dataset"
NUM_EPOCHS=10
RUN_ID="run_3"
BATCH_SIZE=64
LEARNING_RATE=1e-2

DISPLAY_NAME="${TASK_ID}_${RUN_ID}_$(date +'%Y_%m_%dT%H_%M')"
COMMIT_HASH=$(git rev-parse --short HEAD)

if [[ -z "${BUCKET}" ]]; then
  echo "Bucket env variable not set, exiting"
  exit(0)
else
  BASE_DIR="/gcs/${BUCKET}"
  TASK_DIR="${BASE_DIR}/${TASK_ID}"
  TRAINING_DATA_DIR="${BASE_DIR}/dataset_v1/"
fi

# TODO: remove this. Using prebuild image until pushing to eu gcr is possible
if [[ -z "${CONTAINER_IMAGE_URI}" ]]; then
  echo "Container image uri env set, exiting"
  exit(0)
else

fi


echo "staring training job with args"
echo "TASK_ID: $TASK_ID"
echo "TASK_DIR: $TASK_DIR"
echo "TRAINING_DATA_DIR: $TRAINING_DATA_DIR"
echo "CONTAINER_IMAGE_URI: $CONTAINER_IMAGE_URI"
echo "DISPLAY_NAME: $DISPLAY_NAME"
echo "COMMIT_HASH: $COMMIT_HASH"
# https://cloud.google.com/sdk/gcloud/reference/ai/custom-jobs/create

gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=${DISPLAY_NAME} \
  --worker-pool-spec=machine-type=${MACHINE_TYPE},replica-count=${REPLICA_COUNT},container-image-uri=${CONTAINER_IMAGE_URI},accelerator-type=${GPU_CARD},accelerator-count=${GPU_COUNT} \
  --args=--num-epochs=${NUM_EPOCHS},--task-id=${TASK_DIR},--run-id=${RUN_ID},--training-data-dir=${TRAINING_DATA_DIR},--reuse-epoch,--commit-hash=${COMMIT_HASH},--batch-size=${BATCH_SIZE},--learning-rate=${LEARNING_RATE}
#  #  --worker-pool-spec=machine-type=${MACHINE_TYPE},replica-count=${REPLICA_COUNT},executor-image-uri=${EXECUTOR_IMAGE_URI},local-package-path=${WORKING_DIRECTORY},python-module=${PYTHON_MODULE} \

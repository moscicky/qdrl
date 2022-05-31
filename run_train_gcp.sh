EXECUTOR_IMAGE_URI="europe-docker.pkg.dev/vertex-ai/training/pytorch-xla.1-11:latest"
REGION="europe-west3"
DISPLAY_NAME="ebr_$(date +'%Y-%m-%dT%H-%M-%S')"
PYTHON_MODULE="qdrl.main_train"
WORKING_DIRECTORY="."
MACHINE_TYPE="e2-standard-4"
REPLICA_COUNT=1
NUM_EPOCHS=10

if [[ -z "${BUCKET}" ]]; then
  echo "Bucket env variable not set, exiting"
  exit(0)
else
  TRAINING_DATA_DIR="/gcs/${BUCKET}/dataset_v1"
  JOB_DIR="/gcs/${BUCKET}/training_jobs"
fi

# TODO: uncomment. Use autopackaging if pushing to eu gcr is possible
#if [[ -z "${PROJECT}" ]]; then
#  echo "Project env variable not set, exiting"
#  exit(0)
#else
#  CONTAINER_URI="eu.gcr.io/${PROJECT}/${DISPLAY_NAME}"
#fi

# TODO: remove this. Using prebuild image until pushing to eu gcr is possible
if [[ -z "${CONTAINER_IMAGE_URI}" ]]; then
  echo "Project env variable not set, exiting"
  exit(0)
else

fi

TRAINING_DATA_FILE="dataset.csv"

echo "staring training job with args"
echo "TRAINING_DATA_DIR: $TRAINING_DATA_DIR"
echo "JOB_DIR: $JOB_DIR"
#echo "CONTAINER_URI: $CONTAINER_URI"
echo "CONTAINER_IMAGE_URI: $CONTAINER_IMAGE_URI"
# https://cloud.google.com/sdk/gcloud/reference/ai/custom-jobs/create

gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=${DISPLAY_NAME} \
  --worker-pool-spec=machine-type=${MACHINE_TYPE},replica-count=${REPLICA_COUNT},container-image-uri=${CONTAINER_IMAGE_URI} \
  --args=num-epochs=${NUM_EPOCHS},job-dir=${JOB_DIR},training-data-dir=${TRAINING_DATA_DIR},training-data-file=${TRAINING_DATA_FILE}
  #  --worker-pool-spec=machine-type=${MACHINE_TYPE},replica-count=${REPLICA_COUNT},executor-image-uri=${EXECUTOR_IMAGE_URI},local-package-path=${WORKING_DIRECTORY},python-module=${PYTHON_MODULE} \

#  --reuse-job-dir \
#  --reuse-epoch

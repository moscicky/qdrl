EXECUTOR_IMAGE_URI="europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-11:latest"
WORKING_DIRECTORY="."
PYTHON_MODULE="qdrl.main_train"
NAME="batch_softmax_temperature"
RUN_ID="$(date +'%Y-%m-%d-%H-%M')"

DISPLAY_NAME="${NAME}_${RUN_ID}"
EXTRA_DIRS="datasets/docker,datasets/docker_parquet,configs"
CONFIG="docker_multi_modal_batch_softmax.yml"

# build image which will be pushed to gcr
if [[ -z "${PROJECT}" ]]; then
  echo "Project env variable not set, exiting"
  exit(0)
else
  OUTPUT_IMAGE_URI="eu.gcr.io/${PROJECT}/ebr/${DISPLAY_NAME}"
fi

COMMIT_HASH=$(git rev-parse --short HEAD)

echo "COMMIT_HASH: ${COMMIT_HASH}"

gcloud ai custom-jobs local-run \
  --executor-image-uri=$EXECUTOR_IMAGE_URI \
  --local-package-path=$WORKING_DIRECTORY \
  --python-module=$PYTHON_MODULE \
  --output-image-uri=$OUTPUT_IMAGE_URI \
  --extra-dirs=$EXTRA_DIRS \
  -- \
  --config-file-path=configs/${CONFIG} \
  --commit-hash=${COMMIT_HASH}

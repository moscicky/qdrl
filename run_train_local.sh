EXECUTOR_IMAGE_URI="europe-docker.pkg.dev/vertex-ai/training/pytorch-xla.1-11:latest"
WORKING_DIRECTORY="."
PYTHON_MODULE="qdrl.main_train"
RUN_ID="$(date +'%Y-%m-%d-%H-%M-%S')"
#RUN_ID="..."

TASK_ID="gpu_test"
DISPLAY_NAME="${TASK_ID}_${RUN_ID}"
EXTRA_DIRS="datasets,bucket"


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
  --task-id=$TASK_ID \
  --run-id=$RUN_ID \
  --num-epochs=3 \
  --training-data-dir=datasets/small \
  --commit-hash=$COMMIT_HASH \
  --reuse-epoch

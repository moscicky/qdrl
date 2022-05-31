EXECUTOR_IMAGE_URI="europe-docker.pkg.dev/vertex-ai/training/pytorch-xla.1-11:latest"
WORKING_DIRECTORY="."
PYTHON_MODULE="qdrl.main_train"
OUTPUT_IMAGE_NAME="train_v1"
EXTRA_DIRS="datasets,models,checkpoints"

gcloud ai custom-jobs local-run \
  --executor-image-uri=$EXECUTOR_IMAGE_URI \
  --local-package-path=$WORKING_DIRECTORY \
  --python-module=$PYTHON_MODULE \
  --output-image-uri=$OUTPUT_IMAGE_NAME \
  --extra-dirs=$EXTRA_DIRS \
  -- \
  --num-epochs=10 \
  --job-dir=$WORKING_DIRECTORY \
  --training-data-dir=datasets \
  --training-data-file=small.csv \
  --reuse-job-dir \
  --reuse-epoch

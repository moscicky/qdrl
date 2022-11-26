# qdrl

This repository contains training code used for my master thesis titled 
`Joint Multi-Modal Query-Document Representation Learning` which you can read [here](thesis.pdf).

## Training 
Model training is configured by `config.yaml` file with training parameters.

Model training is done on Google Cloud Platform using Vertex AI Training with custom image. Configs, datasets and models
are stored on Google Cloud Storage, gcsfuse is required.

The training flow is the following:
1. Create local training config - `local_config.yaml`
2. Create training docker image using [dedicated script](run_train_local.sh). Point the training 
script to `local_config.yaml`, as well as your GCP project. The script will output your `${image_name}`.
3. Publish the docker image to gcr: `docker push ${image_name}` 
4. Upload the training config to GCS .
5. Run the [gcp training script](run_train_gcp_gpu.sh) with correct `CONTAINER_IMAGE_URI` and `config.yaml` gcs path.

Models checkpoint is saved after each epoch, model from last epoch is saved separately. 

Evaluation metrics - `recall@k` and `mrr@k` are saved and can be visualized on tensorboard. 

Embedding visualization can be optionally turned on if you want to play with it on TB projector.

## Datasets
There are 3 required datasets for training and evaluation (details are in the thesis). 

1. Training dataset - pairs of query, relevant document
2. Evaluation queries dataset (`recall_validation_queries_dataset`) - pairs of query, relevant document id
3. Evaluation documents dataset (`recall_validation_items_dataset`) - candidate pool for evaluation

## Config
[Example config](configs/example.yaml)

Supported training parameters
- task_id  
- run_id
- num_epochs
- dataset_dir
- batch_size
- learning_rate
- reuse_epoch
- dataloader_workers
- dataset - structure of training features
- loss - can be `batch_softmax` or `triplet`
- text_vectorizer - path to the token dictionary and tokenization config (word_unigram, word_bigram, char_trigram + oov)
- model - can be `SimpleTextEncoder`, `TwoTower`, or `MultiModalTwoTower`
- recall_validation - for what 'k' validation should be run and whether to generate dataset with typos

## Acknowledgments 

Research papers can be found in the thesis. For the code part special thanks goes to:

- https://github.com/adambielski/siamese-triplet
- https://gist.github.com/danmelton/183313
- https://stackoverflow.com/a/58144658/7073537

## FAQ

#### 1. The codebase is awful and does not have tests, why?

Best engineering practices do not apply to master thesis, sorry

#### 2. What does 'qdrl' mean?

qdrl stands for **Q**uery **D**ocument **R**epresentation **L**earning

#### 3. No distributed training?

No.

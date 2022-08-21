import json
import os.path
from typing import List, Dict, Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch.nn.functional as F

from qdrl.args import get_args
from qdrl.checkpoints import save_checkpoint
from qdrl.configs import SimilarityMetric
from qdrl.loader import ChunkingDataset
from qdrl.loss_validator import LossValidator
from qdrl.models import SimpleTextEncoder
from qdrl.recall_validator import RecallValidator
from qdrl.triplets import TripletAssembler, BatchNegativeTripletsAssembler


def train(
        device: torch.device,
        epoch_start: int,
        dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.TripletMarginWithDistanceLoss,
        optimizer: optim.Optimizer,
        n_epochs: int,
        checkpoints_path: str,
        triplet_assembler: TripletAssembler,
        tensorboard_writer: SummaryWriter,
        loss_validator: LossValidator,
        recall_validator: Optional[RecallValidator] = None
):
    batch_idx = 0

    for epoch in range(epoch_start, n_epochs):
        model.train()
        epoch_loss = 0.0
        print(f"Starting epoch: {epoch}")
        for batch_idx, batch in enumerate(dataloader):
            anchor, positive, negative = triplet_assembler.generate_triplets(model, batch, device)
            loss = loss_fn(anchor=anchor, positive=positive, negative=negative)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            epoch_loss += batch_loss

            if batch_idx % 10_000 == 0:
                print(f"processed {batch_idx} batches")

        save_checkpoint(epoch, checkpoints_path, model, optimizer)

        average_loss = epoch_loss / (batch_idx if batch_idx else 1)
        print(f"Finished training epoch: {epoch}, total loss: {epoch_loss}, average loss: {average_loss}")

        validation_total_loss, validation_average_loss = loss_validator.validate(model, epoch)

        tensorboard_writer.add_scalar("Loss/train", epoch_loss, epoch)
        tensorboard_writer.add_scalar("Loss/valid", validation_total_loss, epoch)
        tensorboard_writer.add_scalar("AverageLoss/train", average_loss, epoch)
        tensorboard_writer.add_scalar("AverageLoss/valid", validation_average_loss, epoch)

        if recall_validator:
            print("Starting recall validation")
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            recall = recall_validator.validate(model)
            tensorboard_writer.add_scalar("Recall/valid", recall, epoch)


def init_directories(paths: List[str]):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def init_task_dir(task_id: str, run_id: str, meta: Dict):
    if not os.path.isdir(task_id):
        print("Creating task directory...")
        init_directories([task_id])

    run_id_path = os.path.join(task_id, run_id)
    if not os.path.isdir(run_id_path):
        print("Creating run directory")
        init_directories([run_id_path])
        metadata_path = os.path.join(task_id, run_id, "metadata.json")
        with open(metadata_path, 'w') as mf:
            json.dump(meta, mf)


EMBEDDING_DIM = 256
FC_DIM = 128
NUM_EMBEDDINGS = 50000
TEXT_MAX_LENGTH = 10


def dataset_factory(cols: List[str], num_features: int, max_length: int) -> Callable[[str], ChunkingDataset]:
    return lambda p: ChunkingDataset(p, cols=cols, num_features=num_features, max_length=max_length)


def main(
        task_id: str,
        run_id: str,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        reuse_epoch: bool,
        dataset_dir: str,
        meta: Dict,
        dataloader_workers: int
):
    init_task_dir(task_id=task_id, run_id=run_id, meta=meta)

    tensorboard_logdir_path = os.path.join(task_id, "tensorboard", run_id)
    checkpoint_dir_path = os.path.join(task_id, run_id, "checkpoints")
    model_output_dir_path = os.path.join(task_id, run_id, "models")
    model_output_path = os.path.join(model_output_dir_path, "model_weights.pth")
    checkpoints_path = os.path.join(checkpoint_dir_path, "checkpoint")

    dataset_fn = dataset_factory(cols=["query_search_phrase", "product_name"], num_features=NUM_EMBEDDINGS,
                                 max_length=TEXT_MAX_LENGTH)
    training_dataset = dataset_fn(os.path.join(dataset_dir, "training_dataset"))
    validation_dataset = dataset_fn(os.path.join(dataset_dir, "validation_dataset"))

    triplet_assembler = BatchNegativeTripletsAssembler(batch_size=batch_size, negatives_count=batch_size - 1)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, num_workers=dataloader_workers, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=dataloader_workers, drop_last=True)

    layout = {
        "metrics": {
            "loss": ["Multiline", ["Loss/train", "Loss/valid"]],
            "averageLoss": ["Multiline", ["AverageLoss/train", "AverageLoss/valid"]],
        },
    }

    tensorboard_writer = SummaryWriter(log_dir=tensorboard_logdir_path)

    tensorboard_writer.add_custom_scalars(layout)

    model = SimpleTextEncoder(num_embeddings=NUM_EMBEDDINGS, embedding_dim=EMBEDDING_DIM, fc_dim=FC_DIM,
                              output_dim=FC_DIM)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))

    epoch_start = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Will train using device: {device}")
    model.to(device)

    if os.path.exists(checkpoints_path):
        print("Checkpoint found, trying to resume training...")
        checkpoint = torch.load(checkpoints_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if reuse_epoch:
            epoch_start = checkpoint['epoch']
        print(f"Model loaded, will resume from epoch: {epoch_start}...")
    else:
        print("Checkpoint not found, will create training directories from scratch...")
        init_directories([tensorboard_logdir_path, model_output_dir_path, checkpoint_dir_path])

    validator = LossValidator(
        dataloader=validation_dataloader,
        loss_fn=triplet_loss,
        triplet_assembler=triplet_assembler,
        device=device
    )

    recall_validator = RecallValidator(
        candidates_path=os.path.join(dataset_dir, "recall_validation_items_dataset", "items.json"),
        queries_path=os.path.join(dataset_dir, "recall_validation_queries_dataset", "queries.json"),
        num_embeddings=NUM_EMBEDDINGS,
        text_max_length=TEXT_MAX_LENGTH,
        embedding_dim=FC_DIM,
        similarity_metric=SimilarityMetric.COSINE,
        embedding_batch_size=4096,
        k=1024,
        query_batch_size=128,
        device=device
    )

    train(
        device=device,
        epoch_start=epoch_start,
        dataloader=training_dataloader,
        model=model,
        loss_fn=triplet_loss,
        optimizer=optimizer,
        n_epochs=num_epochs,
        checkpoints_path=checkpoints_path,
        triplet_assembler=triplet_assembler,
        tensorboard_writer=tensorboard_writer,
        loss_validator=validator,
        recall_validator=recall_validator if args.validate_recall else None
    )
    print("Training finished, saving the model from last epoch...")

    tensorboard_writer.flush()
    tensorboard_writer.close()

    torch.save(model.state_dict(), model_output_path)
    print("Model saved successfully, exiting...")


if __name__ == '__main__':
    args = get_args()
    print(f"Starting training job with args: {args}")

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    main(
        num_epochs=args.num_epochs,
        task_id=args.task_id,
        run_id=args.run_id,
        batch_size=args.batch_size,
        dataset_dir=args.dataset_dir,
        learning_rate=args.learning_rate,
        reuse_epoch=args.reuse_epoch,
        meta=vars(args),
        dataloader_workers=args.dataloader_workers
    )

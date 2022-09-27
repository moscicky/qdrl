import os.path
from typing import List, Callable, Optional

import torch
from omegaconf import OmegaConf, DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim

from qdrl.args import get_args
from qdrl.checkpoints import save_checkpoint
from qdrl.loader import ChunkingDataset
from qdrl.loss_computer import LossComputer
from qdrl.loss_validator import LossValidator
from qdrl.preprocess import TextVectorizer
from qdrl.recall_validator import RecallValidator, setup_recall_validator
from qdrl.setup import setup_vectorizer, setup_loss, setup_model, parse_features, Features


def train(
        device: torch.device,
        epoch_start: int,
        dataloader: DataLoader,
        model: nn.Module,
        loss_computer: LossComputer,
        optimizer: optim.Optimizer,
        n_epochs: int,
        checkpoints_path: str,
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
            loss = loss_computer.compute(model=model, batch=batch, device=device)
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
            recalls = recall_validator.validate(model)
            for k, recall in recalls.items():
                tensorboard_writer.add_scalar(f"Recall@{k}/valid", recall, epoch)


def init_directories(paths: List[str]):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def init_task_dir(task_id: str, run_id: str, conf: DictConfig):
    if not os.path.isdir(task_id):
        print("Creating task directory...")
        init_directories([task_id])

    run_id_path = os.path.join(task_id, run_id)
    if not os.path.isdir(run_id_path):
        print("Creating run directory")
        init_directories([run_id_path])
        metadata_path = os.path.join(task_id, run_id, "config.yaml")
        with open(metadata_path, 'w') as mf:
            OmegaConf.save(config=conf, f=mf)


def dataset_factory(vectorizer: TextVectorizer, features: Features) -> Callable[[str], ChunkingDataset]:
    return lambda p: ChunkingDataset(p, vectorizer, features)


def main(
        config: DictConfig
):
    init_task_dir(
        task_id=config.task_id,
        run_id=config.run_id,
        conf=config)

    tensorboard_logdir_path = os.path.join(config.task_id, "tensorboard", config.run_id)
    checkpoint_dir_path = os.path.join(config.task_id, config.run_id, "checkpoints")
    model_output_dir_path = os.path.join(config.task_id, config.run_id, "models")
    model_output_path = os.path.join(model_output_dir_path, "model_weights.pth")
    checkpoints_path = os.path.join(checkpoint_dir_path, "checkpoint")

    features = parse_features(config)
    vectorizer = setup_vectorizer(config)

    dataset_fn = dataset_factory(vectorizer=vectorizer, features=features)
    training_dataset = dataset_fn(os.path.join(config.dataset_dir, "training_dataset"))
    validation_dataset = dataset_fn(os.path.join(config.dataset_dir, "validation_dataset"))

    loss_computer = setup_loss(config)
    training_dataloader = DataLoader(training_dataset, batch_size=conf.batch_size, num_workers=conf.dataloader_workers,
                                     drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=conf.batch_size,
                                       num_workers=conf.dataloader_workers,
                                       drop_last=True)

    layout = {
        "metrics": {
            "loss": ["Multiline", ["Loss/train", "Loss/valid"]],
            "averageLoss": ["Multiline", ["AverageLoss/train", "AverageLoss/valid"]],
        },
    }

    tensorboard_writer = SummaryWriter(log_dir=tensorboard_logdir_path)

    tensorboard_writer.add_custom_scalars(layout)

    model = setup_model(config)

    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate)

    epoch_start = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Will train using device: {device}")
    model.to(device)

    if os.path.exists(checkpoints_path):
        print("Checkpoint found, trying to resume training...")
        checkpoint = torch.load(checkpoints_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if conf.reuse_epoch:
            epoch_start = checkpoint['epoch']
        print(f"Model loaded, will resume from epoch: {epoch_start}...")
    else:
        print("Checkpoint not found, will create training directories from scratch...")
        init_directories([tensorboard_logdir_path, model_output_dir_path, checkpoint_dir_path])

    validator = LossValidator(
        dataloader=validation_dataloader,
        loss_computer=loss_computer,
        device=device
    )

    recall_validator = setup_recall_validator(
        config=conf,
        vectorizer=vectorizer,
        device=device,
        features=features
    )

    train(
        device=device,
        epoch_start=epoch_start,
        dataloader=training_dataloader,
        model=model,
        loss_computer=loss_computer,
        optimizer=optimizer,
        n_epochs=conf.num_epochs,
        checkpoints_path=checkpoints_path,
        tensorboard_writer=tensorboard_writer,
        loss_validator=validator,
        recall_validator=recall_validator
    )
    print("Training finished, saving the model from last epoch...")

    tensorboard_writer.flush()
    tensorboard_writer.close()

    torch.save(model.state_dict(), model_output_path)
    print("Model saved successfully, exiting...")


if __name__ == '__main__':
    args = get_args()
    config_file = args.config_file_path
    conf = OmegaConf.load(config_file)
    if args.commit_hash:
        conf.commit_hash = args.commit_hash
    print(f"Starting training job with args: {OmegaConf.to_yaml(conf)}")

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    main(conf)

import argparse
import os.path
from argparse import ArgumentParser
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from qdrl.loader import TripletsDataset


def train(
        epoch_start: int,
        dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.TripletMarginWithDistanceLoss,
        optimizer: optim.Optimizer,
        n_epochs: int,
        checkpoints_path: str):
    model.train()
    for epoch in range(epoch_start, n_epochs):
        epoch_loss = 0.0
        print(f"Starting epoch: {epoch}")
        for batch_idx, batch in enumerate(dataloader):
            anchor, positive, negative = batch

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = loss_fn(anchor=anchor_out, positive=positive_out, negative=negative_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        save_checkpoint(epoch, checkpoints_path, model, optimizer)

        print(f"Finished epoch: {epoch}, loss: {epoch_loss}")


class NeuralNet(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(NeuralNet, self).__init__()
        self.embedding_layer = nn.EmbeddingBag(num_embeddings=num_embeddings,
                                               embedding_dim=embedding_dim,
                                               mode="mean",
                                               padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim // 2)
        )

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        return self.embedding_layer(text)


EMBEDDING_DIM = 128
NUM_EMBEDDINGS = 50000


def save_checkpoint(epoch: int, path: str, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def get_args():
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        '--num-epochs',
        type=int,
        required=True
    )

    args_parser.add_argument(
        '--job-dir',
        type=str,
        required=True
    )

    args_parser.add_argument(
        '--reuse-job-dir',
        action='store_true',
        default=False,
    )

    args_parser.add_argument(
        '--training-data-dir',
        type=str,
        required=True,
    )

    args_parser.add_argument(
        '--training-data-file',
        type=str,
        default=None
    )

    args_parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3
    )

    args_parser.add_argument(
        '--reuse-epoch',
        action='store_true',
        default=False,
    )

    return args_parser.parse_args()


def main(
        num_epochs: int,
        learning_rate: float,
        job_dir: str,
        reuse_job_dir: bool,
        reuse_epoch: bool,
        training_data_dir: str,
        training_data_file: Optional[str]
):
    model_output_path = os.path.join(job_dir, "models", "model_weights.pth")
    checkpoints_path = os.path.join(job_dir, "checkpoints", "checkpoint")
    if training_data_file:
        dataset_path = os.path.join(training_data_dir, training_data_file)
    else:
        dataset_path = training_data_dir

    dataset = TripletsDataset(dataset_path, num_features=NUM_EMBEDDINGS)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=0, shuffle=True)

    model = NeuralNet(num_embeddings=NUM_EMBEDDINGS, embedding_dim=EMBEDDING_DIM)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))

    epoch_start = 0

    if reuse_job_dir:
        print('Trying to resume training from checkpoint...')
        checkpoint = torch.load(checkpoints_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if reuse_epoch:
            epoch_start = checkpoint['epoch']

    train(
        epoch_start=epoch_start,
        dataloader=dataloader,
        model=model,
        loss_fn=triplet_loss,
        optimizer=optimizer,
        n_epochs=num_epochs,
        checkpoints_path=checkpoints_path)

    torch.save(model.state_dict(), model_output_path)


if __name__ == '__main__':
    is_ide = False

    if is_ide:
        main(
            num_epochs=10,
            job_dir='.',
            reuse_job_dir=True,
            training_data_dir='datasets',
            training_data_file='small.csv',
            learning_rate=1e-3,
            reuse_epoch=False
        )
    else:
        args = get_args()
        print(f"Starting training job with args: {args}")

        main(
            num_epochs=args.num_epochs,
            job_dir=args.job_dir,
            reuse_job_dir=args.reuse_job_dir,
            training_data_dir=args.training_data_dir,
            training_data_file=args.training_data_file,
            learning_rate=args.learning_rate,
            reuse_epoch=args.reuse_epoch
        )

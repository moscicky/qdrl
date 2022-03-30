import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from loader import QueryDocumentDataset
from config import load_config


def batch_negative_triplets(
        anchor: torch.Tensor,
        positives: torch.Tensor,
        batch_size: int) -> [torch.Tensor,
                             torch.Tensor,
                             torch.Tensor]:
    negatives = torch.roll(positives, 1, 0)
    for i in range(2, batch_size):
        negatives = torch.vstack([negatives, torch.roll(positives, 1, dims=0)])
    anchor = anchor.repeat(batch_size - 1, 1)
    positives = positives.repeat(batch_size - 1, 1)
    return anchor, positives, negatives


def train(
        dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.TripletMarginWithDistanceLoss,
        optimizer: optim.Optimizer,
        n_epochs: int):
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            anchor, positive = batch
            anchor_out = model(anchor)
            positive_out = model(positive)
            a, p, n = anchor_out, positive_out, torch.randn((len(anchor_out), EMBEDDING_DIM // 2))
            if batch_idx == 0:
                print(a, p, n)

            loss = loss_fn(anchor=a, positive=p, negative=n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch: {epoch}, loss: {epoch_loss}")


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
        return self.fc(self.embedding_layer(text))


EMBEDDING_DIM = 10

if __name__ == '__main__':
    config_path = 'resources/configs/training.yml'
    config = load_config(config_path)

    dataset = QueryDocumentDataset(config)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=0)

    model = NeuralNet(num_embeddings=100, embedding_dim=EMBEDDING_DIM)

    n_epochs = 10

    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))

    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(dataloader, model, triplet_loss, optimizer, n_epochs)

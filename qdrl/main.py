import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from loader import QueryDocumentDataset
from config import load_config, TrainingConfig


def train(dataloader: DataLoader, model: nn.Module, loss_fn: nn.TripletMarginWithDistanceLoss,
          optimizer: optim.Optimizer, n_epochs: int):
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx == 0:
                print(batch)


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


if __name__ == '__main__':
    config_path = 'resources/configs/training.yml'
    config = load_config(config_path)

    dataset = QueryDocumentDataset(config)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

    model = NeuralNet(num_embeddings=100, embedding_dim=100)

    n_epochs = 10

    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))

    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(dataloader, model, triplet_loss, optimizer, n_epochs)

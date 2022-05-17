import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from loader import TripletsDataset


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
            anchor, positive, negative = batch

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = loss_fn(anchor=anchor_out, positive=positive_out, negative=negative_out)
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
        return self.embedding_layer(text)


EMBEDDING_DIM = 128
NUM_EMBEDDINGS = 50000

if __name__ == '__main__':

    dataset = TripletsDataset('datasets/small.csv', num_features=NUM_EMBEDDINGS)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=0, shuffle=True)

    model = NeuralNet(num_embeddings=NUM_EMBEDDINGS, embedding_dim=EMBEDDING_DIM)

    n_epochs = 20

    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))

    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(dataloader, model, triplet_loss, optimizer, n_epochs)

    torch.save(model.state_dict(), "models/model_weights.pth")

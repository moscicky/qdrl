import torch
from torch import nn


class LinearEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, output_dim: int):
        super(LinearEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.EmbeddingBag(num_embeddings=num_embeddings,
                            embedding_dim=embedding_dim,
                            mode="mean"),
            nn.Linear(in_features=embedding_dim, out_features=output_dim)
        )

    def forward(self, vectorized_text: torch.Tensor) -> torch.Tensor:
        return self.net(vectorized_text)


class SimpleTextEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, fc_dim: int, output_dim: int):
        super(SimpleTextEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.EmbeddingBag(num_embeddings=num_embeddings,
                            embedding_dim=embedding_dim,
                            mode="mean",
                            padding_idx=0),
            nn.Linear(in_features=embedding_dim, out_features=fc_dim),
            nn.ReLU(),
            nn.Linear(in_features=fc_dim, out_features=output_dim)
        )

    def forward(self, vectorized_text: torch.Tensor) -> torch.Tensor:
        return self.net(vectorized_text)


class RegularizedSimpleTextEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, fc_dim: int, output_dim: int, dropout: float = 0.5):
        super(RegularizedSimpleTextEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.EmbeddingBag(num_embeddings=num_embeddings,
                            embedding_dim=embedding_dim,
                            mode="mean",
                            padding_idx=0),
            nn.Dropout(dropout),
            nn.Linear(in_features=embedding_dim, out_features=fc_dim),
            nn.ReLU(),
            nn.Linear(in_features=fc_dim, out_features=output_dim)
        )

    def forward(self, vectorized_text: torch.Tensor) -> torch.Tensor:
        return self.net(vectorized_text)

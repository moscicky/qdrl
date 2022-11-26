from typing import List

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
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 fc_dim: int,
                 output_dim: int,
                 query_text_feature: str,
                 document_text_feature: str
                 ):
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
        self.query_text_feature = query_text_feature
        self.document_text_feature = document_text_feature

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


class TwoTower(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 text_embedding_dim: int,
                 output_dim: int,
                 hidden_layers: List[int],
                 last_linear: bool,
                 query_text_feature: str,
                 document_text_feature: str):
        super(TwoTower, self).__init__()

        self.text_embedding = nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=text_embedding_dim,
            mode="mean",
            padding_idx=0
        )

        self.query_tower = nn.Sequential(
            self.text_embedding,
            _mlp(input_dim=text_embedding_dim, output_dim=output_dim, hidden_layers=hidden_layers,
                 last_linear=last_linear)
        )

        self.document_tower = nn.Sequential(
            self.text_embedding,
            _mlp(input_dim=text_embedding_dim, output_dim=output_dim, hidden_layers=hidden_layers,
                 last_linear=last_linear)
        )

        self.query_text_feature = query_text_feature
        self.document_text_feature = document_text_feature

    def forward_query(self, text: torch.Tensor) -> torch.Tensor:
        return self.query_tower(text)

    def forward_document(self, text: torch.Tensor) -> torch.Tensor:
        return self.document_tower(text)


def _mlp(input_dim: int, output_dim: int, hidden_layers: List[int], last_linear: bool) -> nn.Module:
    mlp = nn.Sequential()
    layers = [input_dim] + hidden_layers + [output_dim]
    for i in range(1, len(layers)):
        if i == len(layers) - 1:
            if last_linear:
                mlp.append(
                    nn.Linear(in_features=layers[i - 1], out_features=layers[i])
                )
        else:
            mlp.append(
                nn.Linear(in_features=layers[i - 1], out_features=layers[i])
            )
            if len(hidden_layers) > 0:
                mlp.append(
                    nn.ReLU()
                )
    return mlp


class MultiModalTwoTower(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 text_embedding_dim: int,
                 category_embedding_dim: int,
                 category_num_embeddings: int,
                 hidden_layers: List[int],
                 last_linear: bool,
                 output_dim: int,
                 query_text_feature: str,
                 document_text_feature: str,
                 document_categorical_feature: str
                 ):
        super(MultiModalTwoTower, self).__init__()
        self.query_text_feature = query_text_feature
        self.document_text_feature = document_text_feature
        self.document_categorical_feature = document_categorical_feature

        self.text_embedding = nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=text_embedding_dim,
            mode="mean",
            padding_idx=0
        )

        self.query_tower = nn.Sequential(
            self.text_embedding,
            _mlp(input_dim=text_embedding_dim, hidden_layers=hidden_layers, output_dim=output_dim,
                 last_linear=last_linear)
        )

        self.category_embedding = nn.Embedding(
            num_embeddings=category_num_embeddings,
            embedding_dim=category_embedding_dim
        )

        self.document_mlp = _mlp(input_dim=text_embedding_dim + category_embedding_dim,
                                hidden_layers=hidden_layers,
                                output_dim=output_dim, last_linear=last_linear)

    def forward_query(self, text: torch.Tensor) -> torch.Tensor:
        return self.query_tower(text)

    def forward_document(self, text: torch.Tensor, category: torch.Tensor) -> torch.Tensor:
        t = self.text_embedding(text)
        c = self.category_embedding(category)

        combined = torch.cat((t, c), 1)

        return self.document_mlp(combined)

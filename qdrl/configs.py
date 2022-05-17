from enum import Enum
from typing import NamedTuple, List

import numpy as np


class WordVectorizerConfig(NamedTuple):
    max_length: int
    num_features: int


class SimilarityMetric(Enum):
    COSINE = 1
    INNER_PRODUCT = 2


class ModelConfig(NamedTuple):
    num_embeddings: int
    embedding_dim: int


class Item(NamedTuple):
    business_id: str
    text: str


class Query(NamedTuple):
    text: str
    relevant_business_item_ids: List[str]


class QueryEmbedded(NamedTuple):
    text: str
    relevant_aux_ids: List[int]
    embedding: np.ndarray


class EmbeddedItem(NamedTuple):
    business_id: str
    text: str
    embedding: np.ndarray
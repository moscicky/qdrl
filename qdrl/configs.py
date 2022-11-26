import dataclasses
from enum import Enum
from typing import List

from qdrl.preprocess import CategoricalFeatureMapper


class SimilarityMetric(Enum):
    COSINE = 1
    INNER_PRODUCT = 2

@dataclasses.dataclass
class CategoricalFeature:
    name: str
    mapper: CategoricalFeatureMapper


@dataclasses.dataclass
class Features:
    document_features: List[str]
    query_features: List[str]
    text_features: List[str]
    categorical_features: List[CategoricalFeature]
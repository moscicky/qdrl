import typing
from enum import Enum

from qdrl.preprocess import CategoricalFeatureMapper


class SimilarityMetric(Enum):
    COSINE = 1
    INNER_PRODUCT = 2


class CategoricalFeature:
    name: str
    mapper: CategoricalFeatureMapper
    mappings: typing.Dict[str, int]

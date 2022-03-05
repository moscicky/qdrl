import abc
from typing import NamedTuple, List, Dict

from omegaconf import OmegaConf


class DatasetLocations(NamedTuple):
    qd_pairs_path: str
    documents_path: str


class FeatureType(abc.ABC):
    pass


class Feature(NamedTuple):
    name: str
    type: FeatureType


class TextFeature(FeatureType):
    def __init__(self, name: str, embedding_dim: int, num_embeddings: int, max_length: int, type: str):
        self.name = name
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.max_length = max_length
        self.type = type


class TrainingConfig(NamedTuple):
    dataset_location: DatasetLocations
    feature_types: Dict[str, List]
    query_features: List[str]
    document_features: List[str]
    features: List[Feature]


def parse_feature_type(feature_type_config: Dict):
    if feature_type_config["type"] == "text":
        return TextFeature(
            name=feature_type_config["name"],
            type=feature_type_config["type"],
            embedding_dim=feature_type_config["embedding_dim"],
            num_embeddings=feature_type_config["num_embeddings"],
            max_length=feature_type_config["max_length"]
        )
    else:
        raise ValueError(f"Unknown feature type: {feature_type_config}")


def load_config(config_path: str) -> TrainingConfig:
    conf = OmegaConf.load(config_path)
    training_config = conf["training"]
    dataset_location = DatasetLocations(
        training_config["qd_pairs_path"], training_config["documents_path"]
    )
    features_config = conf["features"]

    feature_types = {feature_type["name"]: parse_feature_type(feature_type) for feature_type in
                     features_config["types"]}

    all_features = [Feature(feature["name"], feature_types[feature["type"]]) for feature in
                    features_config["query"] + features_config["document"]]

    query_features = [f["name"] for f in features_config["query"]]
    document_features = [f["name"] for f in features_config["document"]]

    return TrainingConfig(
        dataset_location=dataset_location,
        feature_types=feature_types,
        query_features=query_features,
        document_features=document_features,
        features=all_features
    )

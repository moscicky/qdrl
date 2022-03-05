import json
from typing import Iterator, List, Dict, NamedTuple, Any

import pandas as pd
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data.dataset import T_co
from qdrl.config import TrainingConfig, Feature, TextFeature
from qdrl.preprocess import vectorize


class QueryDocumentDataset(IterableDataset):
    def __init__(self, config: TrainingConfig):
        self.config: TrainingConfig = config
        self.df = self.load_from_file(config.dataset_location.qd_pairs_path)

    def __iter__(self) -> Iterator[T_co]:
        pass

    def __getitem__(self, index) -> T_co:
        pass

    def load_from_file(self, path: str) -> pd.DataFrame:
        with open(path, "rb") as f:
            df = pd.read_csv(f)
            for feature in self.config.features:
                if isinstance(feature.type, TextFeature):
                    df[feature.name] = df[feature.name].map(
                        lambda k: vectorize(k, num_features=feature.type.embeddings_dim,
                                            max_size=feature.type.max_size))
            return df

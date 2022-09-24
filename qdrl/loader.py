import csv
import os
from typing import List, Callable, Optional

import pandas as pd
import torch
from torch.utils.data.dataset import T_co, Dataset, IterableDataset

from qdrl.configs import CategoricalFeature
from qdrl.preprocess import TextVectorizer, clean_phrase


def _read_jsonl(path: str, cols: List[str]) -> pd.DataFrame:
    try:
        with open(path, "rb") as f:
            df = pd.read_json(f, lines=True)
            return df[cols]
    except Exception as e:
        print(f"Failed to load file: {path}, error: {e}, skipping")
        return pd.DataFrame(columns=cols)


def _read_parquet(path: str, cols: List[str]) -> pd.DataFrame:
    try:
        with open(path, "rb") as f:
            df = pd.read_parquet(f, columns=cols)
            return df
    except Exception as e:
        print(f"Failed to load file: {path}, error: {e}, skipping")
        return pd.DataFrame(columns=cols)


def _read_csv(path: str, cols: List[str]) -> pd.DataFrame:
    try:
        with open(path, "rb") as f:
            df = pd.read_csv(f, header=True)
            return df[cols]
    except Exception as e:
        print(f"Failed to load file: {path}, error: {e}, skipping")
        return pd.DataFrame(columns=cols)


def _prepare_dataset(path: str,
                     cols: List[str],
                     type: str,
                     vectorizer: TextVectorizer,
                     text_cols: List[str],
                     categorical_feature: Optional[CategoricalFeature]) -> pd.DataFrame:
    if type == 'csv':
        reader = _read_csv
    elif type == 'json':
        reader = _read_jsonl
    elif type == "parquet":
        reader = _read_parquet
    else:
        raise ValueError(f"Unknown type: {type}")
    print(f"Reading dataset: {path}")
    df = reader(path, cols).astype('str')
    df[text_cols] = df[text_cols].applymap(lambda c: clean_phrase(c))
    df[text_cols] = df[text_cols].applymap(lambda c: vectorizer.vectorize(c))
    if categorical_feature:
        df[[categorical_feature.name]] = df[[categorical_feature]].applymap(lambda c: categorical_feature.mapper.map(c))
    return df


class LazyTextDataset(IterableDataset):
    def __init__(self, path: str,
                 query_features: List[str],
                 product_features: List[str],
                 text_features: List[str],
                 vectorizer: TextVectorizer,
                 categorical_feature: Optional[CategoricalFeature]
                 ):
        self.path = path
        self.vectorizer = vectorizer
        self.query_features = query_features
        self.product_features = product_features
        self.cols = query_features + product_features
        self.text_features = text_features
        self.categorical_feature = categorical_feature

    def __iter__(self):
        ds = _prepare_dataset(self.path,
                              cols=self.cols,
                              type='parquet',
                              vectorizer=self.vectorizer,
                              text_cols=self.text_features,
                              categorical_feature=self.categorical_feature
                              )
        for row in ds.iterrows():
            yield {
                "query": {f: torch.tensor(row[1][f]) for f in self.query_features},
                "product": {f: torch.tensor(row[1][f]) for f in self.product_features}
            }


class ItemsDataset:
    def __init__(self, path: str, cols: List[str], vectorizer: TextVectorizer):
        self.df = _prepare_dataset(path, cols, 'csv', vectorizer)

    def take(self, n: int) -> pd.DataFrame:
        return self.df.sample(n=n)


class ChunkingDataset(IterableDataset):
    def __init__(self, dataset_dir_path: str,
                 query_features: List[str],
                 product_features: List[str],
                 text_features: List[str],
                 vectorizer: TextVectorizer,
                 categorical_feature: Optional[CategoricalFeature]):
        super(ChunkingDataset, self).__init__()
        self.dataset_dir_path = dataset_dir_path
        self.query_features = query_features
        self.product_features = product_features
        self.text_features = text_features
        self.categorical_feature = categorical_feature
        self.vectorizer = vectorizer

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.file_paths()
        worker_files = [f for idx, f in enumerate(files) if idx % worker_info.num_workers == worker_info.id]
        datasets = self._datasets(worker_files)
        print(f"Chunking dataset worker: {worker_info.id}, will use: {len(worker_files)} files")
        for idx, f in enumerate(datasets):
            yield from f

    def _datasets(self, worker_files: List[str]) -> List[LazyTextDataset]:
        dss = []
        for f in worker_files:
            path = os.path.join(self.dataset_dir_path, f)
            dss.append(LazyTextDataset(
                path=path,
                query_features=self.query_features,
                product_features=self.product_features,
                text_features=self.text_features,
                vectorizer=self.vectorizer,
                categorical_feature=self.categorical_feature
            ))
        return dss

    def file_paths(self) -> List[str]:
        files = [f for f in os.listdir(self.dataset_dir_path) if os.path.isfile(os.path.join(self.dataset_dir_path, f))]
        return files

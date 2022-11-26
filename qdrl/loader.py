import os
from typing import List

import pandas as pd
import torch
from torch.utils.data.dataset import T_co, IterableDataset

from qdrl.configs import Features
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
                     type: str,
                     vectorizer: TextVectorizer,
                     features: Features) -> pd.DataFrame:
    if type == 'csv':
        reader = _read_csv
    elif type == 'json':
        reader = _read_jsonl
    elif type == "parquet":
        reader = _read_parquet
    else:
        raise ValueError(f"Unknown type: {type}")
    print(f"Reading dataset: {path}")
    all_features = features.document_features + features.query_features
    df = reader(path, all_features).astype('str')
    df[features.text_features] = df[features.text_features].applymap(lambda c: clean_phrase(c))
    df[features.text_features] = df[features.text_features].applymap(lambda c: vectorizer.vectorize(c))
    for cf in features.categorical_features:
        df[[cf.name]] = df[[cf.name]].applymap(lambda c: cf.mapper.map(c)).astype('int')
    return df


class LazyTextDataset(IterableDataset):
    def __init__(self, path: str,
                 vectorizer: TextVectorizer,
                 features: Features
                 ):
        self.path = path
        self.vectorizer = vectorizer
        self.features = features

    def __iter__(self):
        ds = _prepare_dataset(self.path,
                              type='parquet',
                              vectorizer=self.vectorizer,
                              features=self.features
                              )
        for row in ds.iterrows():
            yield {
                "query": {f: torch.tensor(row[1][f]) for f in self.features.query_features},
                "document": {f: torch.tensor(row[1][f]) for f in self.features.document_features}
            }


class ChunkingDataset(IterableDataset):
    def __init__(self,
                 dataset_dir_path: str,
                 vectorizer: TextVectorizer,
                 features: Features
                 ):
        super(ChunkingDataset, self).__init__()
        self.dataset_dir_path = dataset_dir_path
        self.vectorizer = vectorizer
        self.features = features

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
                vectorizer=self.vectorizer,
                features=self.features
            ))
        return dss

    def file_paths(self) -> List[str]:
        files = [f for f in os.listdir(self.dataset_dir_path) if os.path.isfile(os.path.join(self.dataset_dir_path, f))]
        return files

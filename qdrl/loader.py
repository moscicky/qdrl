import csv
import os
from typing import List

import pandas as pd
import torch
from torch.utils.data.dataset import T_co, Dataset, IterableDataset

from qdrl.preprocess import vectorize_word, clean_phrase


class TripletsDataset(Dataset):
    def __init__(self, path: str, num_features: int, max_length: int = 10):
        self.df = self.transform_df(self._read_csv(path), num_features, max_length)

    @staticmethod
    def transform_df(df: pd.DataFrame, num_features: int, max_length: int) -> pd.DataFrame:
        cols = df[["query_search_phrase", "product_name", "negative_product_name"]].astype('str')
        return cols.applymap(lambda c: vectorize_word(c, num_features=num_features, max_length=max_length))

    def __getitem__(self, idx) -> T_co:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        datum = self.df.iloc[idx]
        return torch.tensor(datum[0]), torch.tensor(datum[1]), torch.tensor(datum[2])

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _read_csv(path: str) -> pd.DataFrame:
        with open(path, "rb") as f:
            df = pd.read_csv(f, sep=",", error_bad_lines=False, warn_bad_lines=True)
            return df


class LazyTextDataset(IterableDataset):
    def __init__(self, path: str, cols: List[str], num_features: int, max_length: int):
        self.path = path
        self.cols = cols
        self.num_features = num_features
        self.max_length = max_length

    def __iter__(self):
        for e in self.prepare_dataset().iterrows():
            yield torch.tensor(e[1][0]), torch.tensor(e[1][1])

    def prepare_dataset(self) -> pd.DataFrame:
        print(f"Reading dataset: {self.path}")
        df = self._read_csv(self.path, self.cols).astype('str')
        df_transformed = df.applymap(
            lambda c: vectorize_word(clean_phrase(c), num_features=self.num_features, max_length=self.max_length))
        return df_transformed

    @staticmethod
    def _read_csv(path: str, cols: List[str]) -> pd.DataFrame:
        try:
            with open(path, "rb") as f:
                df = pd.read_csv(f, sep=",", usecols=cols)
                return df
        except Exception as e:
            print(f"Failed to load file: {path}, error: {e}, skipping")
            return pd.DataFrame(columns=cols)


class ChunkingDataset(IterableDataset):
    def __init__(self, dataset_dir_path: str, cols: List[str], num_features: int, max_length: int):
        super(ChunkingDataset, self).__init__()
        self.dataset_dir_path = dataset_dir_path
        self.cols = cols
        self.num_features = num_features
        self.max_length = max_length

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.file_paths()
        worker_files = [f for idx, f in enumerate(files) if idx % worker_info.num_workers == worker_info.id]
        datasets = [LazyTextDataset(os.path.join(self.dataset_dir_path, f), self.cols, num_features=self.num_features,
                                    max_length=self.max_length) for f in
                    worker_files]
        print(f"Chunking dataset worker: {worker_info.id}, will use: {len(worker_files)} files")
        for idx, f in enumerate(datasets):
            yield from f

    def file_paths(self) -> List[str]:
        files = [f for f in os.listdir(self.dataset_dir_path) if os.path.isfile(os.path.join(self.dataset_dir_path, f))]
        return files
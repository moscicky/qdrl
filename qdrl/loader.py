import csv
import os
from typing import List

import pandas as pd
import torch
from torch.utils.data.dataset import T_co, Dataset, IterableDataset

from qdrl.preprocess import TextVectorizer, clean_phrase


class LazyTextDataset(IterableDataset):
    def __init__(self, path: str, cols: List[str], vectorizer: TextVectorizer):
        self.path = path
        self.cols = cols
        self.vectorizer = vectorizer

    def __iter__(self):
        for e in self.prepare_dataset().iterrows():
            yield torch.tensor(e[1][0]), torch.tensor(e[1][1])

    def prepare_dataset(self) -> pd.DataFrame:
        print(f"Reading dataset: {self.path}")
        df = self._read_csv(self.path, self.cols).astype('str')
        df_cleaned = df.applymap(lambda c: clean_phrase(c))
        df_transformed = df_cleaned.applymap(lambda c: self.vectorizer.vectorize(c))
        return df_transformed

    @staticmethod
    def _read_csv(path: str, cols: List[str]) -> pd.DataFrame:
        try:
            with open(path, "rb") as f:
                df = pd.read_json(f, lines=True)
                return df[cols]
        except Exception as e:
            print(f"Failed to load file: {path}, error: {e}, skipping")
            return pd.DataFrame(columns=cols)


class ChunkingDataset(IterableDataset):
    def __init__(self, dataset_dir_path: str, cols: List[str], vectorizer: TextVectorizer):
        super(ChunkingDataset, self).__init__()
        self.dataset_dir_path = dataset_dir_path
        self.cols = cols
        self.vectorizer = vectorizer

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.file_paths()
        worker_files = [f for idx, f in enumerate(files) if idx % worker_info.num_workers == worker_info.id]
        datasets = [LazyTextDataset(os.path.join(self.dataset_dir_path, f), self.cols, self.vectorizer) for f in
                    worker_files]
        print(f"Chunking dataset worker: {worker_info.id}, will use: {len(worker_files)} files")
        for idx, f in enumerate(datasets):
            yield from f

    def file_paths(self) -> List[str]:
        files = [f for f in os.listdir(self.dataset_dir_path) if os.path.isfile(os.path.join(self.dataset_dir_path, f))]
        return files

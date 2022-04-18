from typing import Iterator

import pandas as pd
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co


class TripletsDataset(IterableDataset):
    def __init__(self, path: str):
        self.df = self._read_csv(path)

    def __iter__(self) -> Iterator[T_co]:
        for i, row in self.df.iterrows():
            yield row

    def __getitem__(self, index) -> T_co:
        pass

    @staticmethod
    def _read_csv(path: str) -> pd.DataFrame:
        with open(path, "rb") as f:
            df = pd.read_csv(f)
            return df

import pandas as pd
import torch
from torch.utils.data.dataset import T_co, Dataset

from qdrl.preprocess import vectorize_word


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
            df = pd.read_csv(f, sep=",")
            return df

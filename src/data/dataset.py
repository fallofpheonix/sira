import torch
import pandas as pd
from torch.utils.data import Dataset, Subset
import numpy as np


class VectorFieldDataset(Dataset):
    def __init__(self, csv_path_or_df):
        if isinstance(csv_path_or_df, pd.DataFrame):
            df = csv_path_or_df
        else:
            df = pd.read_csv(csv_path_or_df)
        self.X = torch.tensor(df[['S', 'I', 'R']].values, dtype=torch.float32)
        self.y = torch.tensor(df[['dS_dt', 'dI_dt', 'dR_dt']].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DatasetSplitter:
    def split(self, dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        n = len(dataset)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n).tolist()
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

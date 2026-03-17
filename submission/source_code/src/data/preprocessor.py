import numpy as np
import pandas as pd


class DataPreprocessor:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._cols = ['S', 'I', 'R', 'dS_dt', 'dI_dt', 'dR_dt']

    def fit(self, df):
        cols = [c for c in self._cols if c in df.columns]
        self.mean_ = df[cols].mean()
        self.std_ = df[cols].std().replace(0, 1)
        return self

    def transform(self, df):
        df = df.copy()
        cols = [c for c in self._cols if c in df.columns]
        df[cols] = (df[cols] - self.mean_[cols]) / self.std_[cols]
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def inverse_transform(self, df):
        df = df.copy()
        cols = [c for c in self._cols if c in df.columns]
        df[cols] = df[cols] * self.std_[cols] + self.mean_[cols]
        return df

import numpy as np
import torch


class OODDetector:
    def __init__(self):
        self.mean_ = None
        self.cov_inv_ = None

    def fit(self, train_data):
        if isinstance(train_data, torch.Tensor):
            train_data = train_data.numpy()
        self.mean_ = np.mean(train_data, axis=0)
        cov = np.cov(train_data.T)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        self.cov_inv_ = np.linalg.pinv(cov)
        return self

    def predict(self, test_data):
        if isinstance(test_data, torch.Tensor):
            test_data = test_data.numpy()
        diff = test_data - self.mean_
        scores = np.array([d @ self.cov_inv_ @ d for d in diff])
        threshold = np.percentile(scores, 95) if len(scores) > 0 else 0
        is_ood = scores > threshold
        return is_ood, scores

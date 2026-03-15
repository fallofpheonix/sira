import numpy as np
import torch


def _to_numpy(data):
    """Convert data to a numpy array, handling torch tensors on any device."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


class OODDetector:
    def __init__(self, threshold_percentile=95):
        """
        Args:
            threshold_percentile: Percentile of in-distribution Mahalanobis
                scores used to set the OOD threshold during fit(). Default 95.
        """
        self.threshold_percentile = threshold_percentile
        self.mean_ = None
        self.cov_inv_ = None
        self.threshold_ = None

    def fit(self, train_data):
        """Fit the detector on in-distribution training data.

        Computes the mean, inverse covariance, and the OOD score threshold
        (``threshold_percentile``-th percentile of training Mahalanobis scores).
        """
        train_data = _to_numpy(train_data)
        self.mean_ = np.mean(train_data, axis=0)
        cov = np.cov(train_data.T)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        self.cov_inv_ = np.linalg.pinv(cov)
        # Calibrate threshold from in-distribution scores
        diff = train_data - self.mean_
        train_scores = np.array([d @ self.cov_inv_ @ d for d in diff])
        self.threshold_ = (
            np.percentile(train_scores, self.threshold_percentile)
            if len(train_scores) > 0
            else 0.0
        )
        return self

    def predict(self, test_data):
        """Detect OOD samples using the threshold calibrated during fit().

        Returns:
            is_ood: boolean array, True where Mahalanobis score > threshold_.
            scores: raw Mahalanobis scores for each sample.
        """
        test_data = _to_numpy(test_data)
        diff = test_data - self.mean_
        scores = np.array([d @ self.cov_inv_ @ d for d in diff])
        is_ood = scores > self.threshold_
        return is_ood, scores

import torch
from pathlib import Path


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self._counter = 0
        self._best = float('inf')

    def __call__(self, val_loss):
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
        else:
            self._counter += 1
        return self._counter >= self.patience


class ModelCheckpoint:
    def __init__(self, path='checkpoints/', monitor='val_loss'):
        self.path = Path(path)
        self.monitor = monitor
        self._best = float('inf')

    def __call__(self, model, epoch, metrics):
        value = metrics.get(self.monitor, float('inf'))
        if value < self._best:
            self._best = value
            self.path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.path / 'best_model.pth')

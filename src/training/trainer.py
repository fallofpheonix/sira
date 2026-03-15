import torch
import torch.nn as nn
from pathlib import Path
from src.training.callbacks import EarlyStopping, ModelCheckpoint


class Trainer:
    def __init__(self, model, optimizer, criterion, device='cpu', config=None,
                 mlflow_enabled=False, model_kwargs=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config or {}
        self.mlflow_enabled = mlflow_enabled
        self.model_kwargs = model_kwargs or {}
        self._mlflow = None
        if mlflow_enabled:
            try:
                import mlflow
                self._mlflow = mlflow
            except ImportError:
                self.mlflow_enabled = False

    def fit(self, train_loader, val_loader=None, epochs=100):
        self.model.to(self.device)
        patience = self.config.get('early_stopping_patience', 20)
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints/')
        early_stopping = EarlyStopping(patience=patience, min_delta=1e-6)
        checkpoint = ModelCheckpoint(path=checkpoint_dir)
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                checkpoint(self.model, epoch, {'val_loss': val_loss})
                if early_stopping(val_loss):
                    break
            if self.mlflow_enabled and self._mlflow:
                self._mlflow.log_metric('train_loss', train_loss, step=epoch)
                if val_loss is not None:
                    self._mlflow.log_metric('val_loss', val_loss, step=epoch)

        return history

    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(X_batch)
            loss = self.criterion(pred, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(train_loader), 1)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred = self.model(X_batch)
                total_loss += self.criterion(pred, y_batch).item()
        return total_loss / max(len(val_loader), 1)

    def test(self, test_loader):
        return self.validate(test_loader)

    def save_checkpoint(self, path, epoch, loss):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'model_name': type(self.model).__name__,
            'model_kwargs': self.model_kwargs,
        }, path)

    def resume_from_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt['epoch']

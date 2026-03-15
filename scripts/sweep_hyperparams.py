#!/usr/bin/env python
"""Hyperparameter sweep.
Usage: python scripts/sweep_hyperparams.py
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SWEEP_CONFIG = {
    'learning_rate': [1e-3, 5e-4, 1e-4],
    'hidden_dim': [64, 128, 256],
    'num_layers': [2, 3, 4],
}


def main():
    data_path = 'data/processed/sir_vector_field.csv'
    if not Path(data_path).exists():
        print(f"Dataset not found at {data_path}. Run generate_data.py first.")
        return

    from src.data.dataset import VectorFieldDataset, DatasetSplitter
    from src.models.architectures.mlp import VectorFieldMLP
    from src.training.trainer import Trainer
    from torch.utils.data import DataLoader

    dataset = VectorFieldDataset(data_path)
    splitter = DatasetSplitter()
    train_ds, val_ds, _ = splitter.split(dataset)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    best_val_loss = float('inf')
    best_config = None

    for lr in SWEEP_CONFIG['learning_rate']:
        for hidden_dim in SWEEP_CONFIG['hidden_dim']:
            model = VectorFieldMLP(hidden_dim=hidden_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            trainer = Trainer(model, optimizer, criterion)
            history = trainer.fit(train_loader, val_loader, epochs=20)
            val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
            print(f"lr={lr}, hidden={hidden_dim} -> val_loss={val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = {'lr': lr, 'hidden_dim': hidden_dim}

    print(f"\nBest config: {best_config} with val_loss={best_val_loss:.6f}")


if __name__ == '__main__':
    main()

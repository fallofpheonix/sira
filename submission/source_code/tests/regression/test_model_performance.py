import pytest
import sys
import torch
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def test_model_performance_threshold(tmp_path):
    from src.data.generator import DataPipeline
    from src.data.dataset import VectorFieldDataset, DatasetSplitter
    from src.models.architectures.mlp import VectorFieldMLP
    from src.training.trainer import Trainer
    from src.training.metrics import compute_r2
    from torch.utils.data import DataLoader

    config = {'population': 100, 'seed': 42}
    pipeline = DataPipeline(config)
    csv_path = tmp_path / "perf_data.csv"
    df = pipeline.run(str(csv_path), num_param_points=20, num_runs_per_param=3,
                      num_timepoints=30)

    dataset = VectorFieldDataset(df)
    splitter = DatasetSplitter()
    train_ds, val_ds, test_ds = splitter.split(dataset, seed=42)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = VectorFieldMLP(hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    trainer = Trainer(model, optimizer, criterion)
    trainer.fit(train_loader, epochs=30)

    test_loader = DataLoader(test_ds, batch_size=32)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds.append(model(X_b))
            targets.append(y_b)
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    r2 = compute_r2(preds, targets)
    # Very loose threshold; even with minimal training the model should
    # produce predictions better than a constant baseline (R² > 0)
    assert r2 > 0.0, f"R2={r2:.3f}: model did not learn useful patterns"

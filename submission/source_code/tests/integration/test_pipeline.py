import pytest
import sys
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def test_full_pipeline(tmp_path):
    from src.data.generator import DataPipeline
    from src.data.dataset import VectorFieldDataset, DatasetSplitter
    from src.models.architectures.mlp import VectorFieldMLP
    from src.training.trainer import Trainer
    from torch.utils.data import DataLoader

    # Generate small dataset
    config = {'population': 100, 'beta_min': 0.2, 'beta_max': 0.5,
              'gamma_min': 0.05, 'gamma_max': 0.2, 'seed': 0}
    pipeline = DataPipeline(config)
    csv_path = tmp_path / "test_data.csv"
    df = pipeline.run(str(csv_path), num_param_points=5, num_runs_per_param=3,
                      num_timepoints=20)
    assert len(df) > 0
    assert csv_path.exists()

    # Create dataset and split
    dataset = VectorFieldDataset(df)
    splitter = DatasetSplitter()
    train_ds, val_ds, test_ds = splitter.split(dataset, seed=0)
    train_loader = DataLoader(train_ds, batch_size=16)
    val_loader = DataLoader(val_ds, batch_size=16)

    # Train for 1 epoch
    model = VectorFieldMLP(hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    trainer = Trainer(model, optimizer, criterion, config={'checkpoint_dir': str(tmp_path / 'checkpoints')})
    history = trainer.fit(train_loader, val_loader, epochs=1)
    assert 'train_loss' in history
    assert len(history['train_loss']) == 1

    # Run inference
    from src.training.metrics import compute_r2, compute_rmse
    model.eval()
    X = torch.tensor(df[['S', 'I', 'R']].values[:10], dtype=torch.float32)
    with torch.no_grad():
        pred = model(X)
    assert pred.shape == (10, 3)

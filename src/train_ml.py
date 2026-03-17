import argparse
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset import VectorFieldDataset, DatasetSplitter
from src.models.architectures.mlp import VectorFieldMLP
from src.symbolic.sindy import SINDy
from src.training.trainer import Trainer


def train(
    data_path,
    model_path,
    batch_size=256,
    lr=1e-3,
    epochs=100,
    hidden=128,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Run generate_data.py first.")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")

    dataset = VectorFieldDataset(df)
    splitter = DatasetSplitter()
    train_ds, val_ds, test_ds = splitter.split(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = VectorFieldMLP(hidden_dim=hidden)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(model, optimizer, criterion)

    print(f"\nTraining VectorFieldMLP: (S,I,R) → (dS/dt,dI/dt,dR/dt)")
    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}  |  Epochs: {epochs}\n")

    trainer.fit(train_loader, val_loader, epochs=epochs)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")

    print("\nSINDy sparse regression:")
    sindy = SINDy().fit(df)
    for eq, expr in sindy.get_equations().items():
        print(f"  {eq} = {expr}")

    print("\nTrue SIR governing equations:")
    print("  dS/dt = -β·S·I")
    print("  dI/dt =  β·S·I - γ·I")
    print("  dR/dt =  γ·I")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train vector-field SIR model.")
    parser.add_argument(
        "--data-path",
        default=str(repo_root / "data" / "processed" / "sir_vector_field.csv"),
    )
    parser.add_argument(
        "--model-path",
        default=str(repo_root / "models" / "vector_field_mlp.pth"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(
        data_path=args.data_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        hidden=args.hidden,
        seed=args.seed,
    )

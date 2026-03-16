import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset import VectorFieldDataset, DatasetSplitter
from src.models.architectures.mlp import VectorFieldMLP
from src.training.trainer import Trainer

# ─────────────────────────────────────────
# SINDy Baseline
# ─────────────────────────────────────────
def sindy_baseline(df):
    """
    SINDy-style least-squares regression with polynomial library.
    Library: [S, I, R, S*I, S*R, I*R, S^2, I^2, R^2]
    """
    S = df['S'].values
    I = df['I'].values
    R = df['R'].values

    # Feature library
    Theta = np.column_stack([
        S, I, R,
        S*I, S*R, I*R,
        S**2, I**2, R**2,
        np.ones(len(S))
    ])

    feature_names = ['S', 'I', 'R', 'SI', 'SR', 'IR', 'S^2', 'I^2', 'R^2', '1']

    results = {}
    for target_col in ['dS_dt', 'dI_dt', 'dR_dt']:
        dy = df[target_col].values
        # Least-squares: minimise ||Theta @ xi - dy||
        xi, _, _, _ = np.linalg.lstsq(Theta, dy, rcond=None)
        terms = [(name, coef) for name, coef in zip(feature_names, xi) if abs(coef) > 1e-3]
        results[target_col] = terms

    return results

# ─────────────────────────────────────────
# Training
# ─────────────────────────────────────────
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
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model     = VectorFieldMLP(hidden_dim=hidden)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(model, optimizer, criterion)

    print(f"\nTraining vector field MLP: (S,I,R) → (dS/dt,dI/dt,dR/dt)")
    print(f"  Train: {len(train_ds):,}  |  Test: {len(test_ds):,}  |  Epochs: {epochs}\n")

    trainer.fit(train_loader, val_loader, epochs=epochs)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(model_path, epochs, 0.0) # Loss is dummy for now
    # Also save just the state dict for backward compatibility if needed, or update consumers
    torch.save(model.state_dict(), model_path.with_suffix('.pth'))
    
    print(f"\nModel saved: {model_path.with_suffix('.pth')}")

    # ── SINDy baseline ──
    print("\n── SINDy Baseline (sparse regression) ──")
    sindy_results = sindy_baseline(df)
    for eq, terms in sindy_results.items():
        expr = " + ".join(f"{c:.4f}*{n}" for n, c in terms)
        print(f"  {eq} = {expr}")

    print("\n── True SIR ODE (ground truth) ──")
    print("  dS/dt = -β * S * I")
    print("  dI/dt =  β * S * I - γ * I")
    print("  dR/dt =  γ * I")

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
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

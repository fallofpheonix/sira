"""
SIRA ML Trainer — Vector Field Learning

Task: (S, I, R) → (dS/dt, dI/dt, dR/dt)
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────
class VectorFieldDataset(Dataset):
    """
    Inputs:  (S, I, R) — normalized state
    Targets: (dS/dt, dI/dt, dR/dt) — time derivatives
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df[['S', 'I', 'R']].values, dtype=torch.float32)
        self.y = torch.tensor(df[['dS_dt', 'dI_dt', 'dR_dt']].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────
class VectorFieldMLP(nn.Module):
    """
    Learns a neural approximation of F(S,I,R) = (dS/dt, dI/dt, dR/dt).
    After training, symbolic regression can extract the functional form.
    """
    def __init__(self, input_dim=3, output_dim=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)

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

    dataset = VectorFieldDataset(data_path)
    n_train = int(0.8 * len(dataset))
    n_test  = len(dataset) - n_train
    split_gen = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=split_gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model     = VectorFieldMLP(hidden=hidden)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining vector field MLP: (S,I,R) → (dS/dt,dI/dt,dR/dt)")
    print(f"  Train: {n_train:,}  |  Test: {n_test:,}  |  Epochs: {epochs}\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for X_t, y_t in test_loader:
                    test_loss += criterion(model(X_t), y_t).item()
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {total_loss/len(train_loader):.6f} | "
                  f"Test Loss:  {test_loss/len(test_loader):.6f}")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")

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

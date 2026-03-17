import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from src.symbolic.sindy import SINDy
from sira.core.paths import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH
from sira.services.training_service import TrainingRunRequest, TrainingService


def train(
    data_path,
    model_path,
    batch_size=256,
    lr=1e-3,
    epochs=100,
    hidden=128,
    seed=42,
):
    np.random.seed(seed)
    data_path = Path(data_path)
    model_path = Path(model_path)
    if not data_path.exists():
        print(f"Dataset not found at {data_path}. Run src/generate_data.py first.")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")

    print(f"\nTraining vector field MLP: (S,I,R) → (dS/dt,dI/dt,dR/dt)")
    service = TrainingService()
    result = service.train_vector_field_model(
        TrainingRunRequest(
            dataset_path=data_path,
            model_output_path=model_path,
            hidden_dim=hidden,
            batch_size=batch_size,
            learning_rate=lr,
            epochs=epochs,
            seed=seed,
        )
    )

    print(f"  Rows: {result.rows_seen:,}  |  Epochs: {epochs}\n")
    print(f"Model saved: {result.model_path}")
    print(f"Checkpoint saved: {result.checkpoint_path}")

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
        default=str(DEFAULT_DATASET_PATH),
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
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

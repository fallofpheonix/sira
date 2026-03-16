import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import torch

from src.models.architectures.mlp import VectorFieldMLP
from src.visualization.plots import plot_parity


def visualize_results(
    data_path,
    model_path,
    output_path,
    seed=0,
    sample_size=5000,
    hidden_dim=128,
    num_layers=3,
    activation='tanh',
    dropout=0.0,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_path = Path(data_path)
    model_path = Path(model_path)

    if not data_path.exists():
        print(f"Dataset not found at {data_path}. Run src/generate_data.py first.")
        return
    if not model_path.exists():
        print(f"Model not found at {model_path}. Run src/train_ml.py first.")
        return

    df = pd.read_csv(data_path)
    if len(df) == 0:
        print("Dataset is empty.")
        return

    sample_size = min(sample_size, len(df))
    df_sample = df.sample(n=sample_size, random_state=seed)

    X = torch.tensor(df_sample[['S', 'I', 'R']].values, dtype=torch.float32)
    y_true = df_sample[['dS_dt', 'dI_dt', 'dR_dt']].values

    model = VectorFieldMLP(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        dropout=dropout,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        y_pred = model(X).numpy()

    # Parity plots
    plot_parity(y_true, y_pred, output_path=output_path)
    print(f"Results plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model and symbolic results.")
    parser.add_argument(
        "--data-path",
        default=str(repo_root / "data" / "processed" / "sir_vector_field.csv"),
    )
    parser.add_argument(
        "--model-path",
        default=str(repo_root / "models" / "vector_field_mlp.pth"),
    )
    parser.add_argument(
        "--output-path",
        default=str(repo_root / "results" / "vector_field_parity.png"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()
    visualize_results(
        data_path=args.data_path,
        model_path=args.model_path,
        output_path=args.output_path,
        seed=args.seed,
        sample_size=args.sample_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        activation=args.activation,
        dropout=args.dropout,
    )

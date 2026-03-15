import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from train_ml import VectorFieldMLP


def visualize_results(data_path, model_path, output_path, seed=0, sample_size=5000):
    np.random.seed(seed)
    torch.manual_seed(seed)

    df = pd.read_csv(data_path)
    if len(df) == 0:
        print("Dataset is empty.")
        return

    sample_size = min(sample_size, len(df))
    df_sample = df.sample(n=sample_size, random_state=seed)

    X = torch.tensor(df_sample[['S', 'I', 'R']].values, dtype=torch.float32)
    y_true = df_sample[['dS_dt', 'dI_dt', 'dR_dt']].values

    model = VectorFieldMLP()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        y_pred = model(X).numpy()

    # Parity plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = ['dS_dt', 'dI_dt', 'dR_dt']
    for i, ax in enumerate(axes):
        ax.scatter(y_true[:, i], y_pred[:, i], s=6, alpha=0.4)
        ax.set_title(f"{labels[i]}: true vs pred")
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Results plot saved to {output_path}")

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
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
    args = parser.parse_args()
    visualize_results(
        data_path=args.data_path,
        model_path=args.model_path,
        output_path=args.output_path,
        seed=args.seed,
        sample_size=args.sample_size,
    )

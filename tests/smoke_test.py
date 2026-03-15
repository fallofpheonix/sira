import argparse
from pathlib import Path

from generate_data import generate_dataset
from train_ml import train
from visualize_results import visualize_results


def main():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Fast smoke test for SIRA pipeline.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-param-points", type=int, default=20)
    parser.add_argument("--num-runs-per-param", type=int, default=5)
    parser.add_argument("--num-timepoints", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    data_path = repo_root / "data" / "processed" / "sir_vector_field.csv"
    model_path = repo_root / "models" / "vector_field_mlp.pth"
    output_path = repo_root / "results" / "vector_field_parity.png"

    generate_dataset(
        output_path=data_path,
        num_param_points=args.num_param_points,
        num_runs_per_param=args.num_runs_per_param,
        num_timepoints=args.num_timepoints,
        seed=args.seed,
    )
    train(
        data_path=data_path,
        model_path=model_path,
        epochs=args.epochs,
        seed=args.seed,
    )
    visualize_results(
        data_path=data_path,
        model_path=model_path,
        output_path=output_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

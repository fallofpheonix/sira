import argparse
from pathlib import Path
from sira.core.paths import DEFAULT_DATASET_PATH
from sira.services.dataset_service import DatasetBuildRequest, DatasetService


def generate_dataset(
    output_path,
    num_param_points=500,
    num_runs_per_param=20,
    population=1000,
    num_timepoints=100,
    beta_min=0.1,
    beta_max=0.9,
    gamma_min=0.02,
    gamma_max=0.4,
    seed=42,
    max_time=150,
):
    service = DatasetService()
    print(f"Generating {num_param_points} parameter points × {num_runs_per_param} runs each...")
    df = service.build_vector_field_dataset(
        DatasetBuildRequest(
            output_path=Path(output_path),
            num_param_points=num_param_points,
            num_runs_per_param=num_runs_per_param,
            population=population,
            num_timepoints=num_timepoints,
            beta_min=beta_min,
            beta_max=beta_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            seed=seed,
            max_time=max_time,
        )
    )
    print(f"\nDataset saved: {output_path}  ({len(df)} rows, {len(df.columns)} columns)")
    print(df.head(3))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate vector-field SIR dataset.")
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_DATASET_PATH),
    )
    parser.add_argument("--num-param-points", type=int, default=500)
    parser.add_argument("--num-runs-per-param", type=int, default=20)
    parser.add_argument("--population", type=int, default=1000)
    parser.add_argument("--num-timepoints", type=int, default=100)
    parser.add_argument("--beta-min", type=float, default=0.1)
    parser.add_argument("--beta-max", type=float, default=0.9)
    parser.add_argument("--gamma-min", type=float, default=0.02)
    parser.add_argument("--gamma-max", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-time", type=float, default=150)
    args = parser.parse_args()
    generate_dataset(
        output_path=args.output_path,
        num_param_points=args.num_param_points,
        num_runs_per_param=args.num_runs_per_param,
        population=args.population,
        num_timepoints=args.num_timepoints,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        seed=args.seed,
        max_time=args.max_time,
    )


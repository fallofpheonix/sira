import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from sira.core.paths import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, DEFAULT_RESULTS_PATH
from sira.services.reporting_service import ReportBuildRequest, ReportingService


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
    data_path = Path(data_path)
    model_path = Path(model_path)
    output_path = Path(output_path)

    service = ReportingService()
    report_path = service.build_parity_report(
        ReportBuildRequest(
            dataset_path=data_path,
            model_path=model_path,
            output_path=output_path,
            seed=seed,
            sample_size=sample_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )
    )
    print(f"Results plot saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model and symbolic results.")
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATASET_PATH),
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_RESULTS_PATH),
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

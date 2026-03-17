#!/usr/bin/env python
"""Full end-to-end experiment runner.
Usage: python scripts/run_experiment.py --config config/base.yaml
"""
import argparse

from _bootstrap import bootstrap

bootstrap()

from sira.services.experiment_service import ExperimentService


def main():
    parser = argparse.ArgumentParser(description="Run SIRA experiment.")
    parser.add_argument('--config', default='config/base.yaml')
    parser.add_argument('--data-config', default='config/data.yaml')
    parser.add_argument('--model-config', default='config/model.yaml')
    args = parser.parse_args()

    service = ExperimentService()
    result = service.run(args.config, args.data_config, args.model_config)
    print(f"Dataset ready: {result.dataset_path}")
    print(f"Model saved: {result.model_path}")
    print(f"Symbolic expressions saved: {result.symbolic_expression_path}")
    print("Experiment complete.")


if __name__ == '__main__':
    main()

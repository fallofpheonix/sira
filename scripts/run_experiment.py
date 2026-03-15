#!/usr/bin/env python
"""Full end-to-end experiment runner.
Usage: python scripts/run_experiment.py --config config/base.yaml
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Run SIRA experiment.")
    parser.add_argument('--config', default='config/base.yaml')
    parser.add_argument('--data-config', default='config/data.yaml')
    parser.add_argument('--model-config', default='config/model.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)['data']
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    seed = base_cfg['experiment'].get('seed', 42)
    set_seed(seed)
    output_dir = Path(base_cfg['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    from src.data.generator import DataPipeline
    data_path = data_cfg.get('output_path', 'data/processed/sir_vector_field.csv')
    print(f"[1/3] Generating dataset -> {data_path}")
    pipeline = DataPipeline(data_cfg)
    df = pipeline.run(
        output_path=data_path,
        num_param_points=data_cfg.get('num_param_points', 50),
        num_runs_per_param=data_cfg.get('num_runs_per_param', 5),
        num_timepoints=data_cfg.get('num_timepoints', 50),
    )
    print(f"Dataset: {len(df)} rows")

    # Train model
    from src.data.dataset import VectorFieldDataset, DatasetSplitter
    from src.models.registry import get_model
    from src.training.trainer import Trainer
    from torch.utils.data import DataLoader

    print("[2/3] Training model...")
    dataset = VectorFieldDataset(df)
    splitter = DatasetSplitter()
    train_ds, val_ds, test_ds = splitter.split(dataset, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=model_cfg['training']['batch_size'],
                              shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=model_cfg['training']['batch_size'])

    model_params = {k: v for k, v in model_cfg['model'].items() if k != 'name'}
    model = get_model(model_cfg['model']['name'], **model_params)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_cfg['training']['learning_rate'],
        weight_decay=model_cfg['training'].get('weight_decay', 0.0),
    )
    criterion = torch.nn.MSELoss()
    trainer = Trainer(model, optimizer, criterion, config=model_cfg['training'])
    history = trainer.fit(train_loader, val_loader, epochs=model_cfg['training']['epochs'])

    model_path = Path('models') / 'vector_field_mlp.pth'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    # Symbolic regression
    print("[3/3] Running SINDy symbolic regression...")
    from src.symbolic.sindy import SINDy
    sindy = SINDy()
    sindy.fit(df)
    eqs = sindy.get_equations()
    for eq, expr in eqs.items():
        print(f"  {eq} = {expr}")
    expr_path = output_dir / 'symbolic_expression.txt'
    sindy.save(expr_path)
    print(f"Symbolic expressions saved: {expr_path}")
    print("Experiment complete.")


if __name__ == '__main__':
    main()

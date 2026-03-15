# SIRA: Stochastic Infection Recovery Analysis

## Overview

SIRA is a modular production-quality ML pipeline that learns the **SIR epidemic vector field**:

- **Input**: (S, I, R) — normalized state
- **Output**: (dS/dt, dI/dt, dR/dt) — time derivatives

from stochastic Gillespie simulations, and then performs symbolic regression to recover the governing equations.

## Pipeline Diagram

```
Gillespie Simulations  →  Ensemble Averaging  →  Derivative Estimation
        ↓
  VectorFieldDataset  →  Trainer (MLP/Neural ODE/Physics-Informed)
        ↓
  Symbolic Discovery (SINDy)  →  results/symbolic_expression.txt
        ↓
  FastAPI Inference Server  →  POST /predict/vector_field
```

## Project Structure

```
SIRA/
├── config/                  # YAML experiment configs
│   ├── base.yaml
│   ├── model.yaml
│   └── data.yaml
├── src/
│   ├── core/                # Simulator + parallel execution
│   ├── data/                # Data pipeline (generator, dataset, preprocessor, validator)
│   ├── models/              # Model architectures (MLP, Neural ODE, Physics-Informed) + registry
│   ├── training/            # Trainer, losses, metrics, callbacks
│   ├── symbolic/            # SINDy, PySR wrapper, expression utilities
│   ├── inference/           # Predictor + FastAPI server
│   ├── evaluation/          # Benchmarks + OOD detection
│   └── visualization/       # Plots + dashboard
├── scripts/                 # Experiment runner, hyperparam sweep, deploy
├── notebooks/               # Jupyter walkthrough notebooks
└── tests/
    ├── unit/
    ├── integration/
    └── regression/
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start (Original Pipeline — Backward Compatible)

```bash
# 1. Generate dataset
python src/generate_data.py

# 2. Train model
python src/train_ml.py

# 3. Visualize results
python src/visualize_results.py
```

## Modular Pipeline

```bash
# Full experiment (data generation + training + symbolic regression)
python scripts/run_experiment.py --config config/base.yaml

# Hyperparameter sweep
python scripts/sweep_hyperparams.py

# Start inference API server
python scripts/deploy_model.py --model-path models/vector_field_mlp.pth
```

## API Usage

After starting the server:

```bash
# Health check
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/models

# Predict vector field at a point
curl -X POST http://localhost:8000/predict/vector_field \
     -H "Content-Type: application/json" \
     -d '{"S": 0.9, "I": 0.05, "R": 0.05}'

# Simulate a trajectory
curl -X POST http://localhost:8000/simulate/trajectory \
     -H "Content-Type: application/json" \
     -d '{"S0": 0.99, "I0": 0.01, "R0": 0.0, "num_steps": 100, "dt": 0.5}'
```

## Testing

```bash
pytest tests/ -v
```

## Model Architectures

| Model | Description |
|-------|-------------|
| `VectorFieldMLP` | 3-layer MLP with Tanh (default) |
| `NeuralODEBaseline` | Euler-integrated ODE approximation |
| `PhysicsInformedMLP` | MLP enforcing dS+dI+dR=0 conservation |

## Technology Stack

- **Language**: Python 3.x
- **ML Framework**: PyTorch
- **API**: FastAPI + Uvicorn
- **Experiment Tracking**: MLflow (optional)
- **Libraries**: NumPy, Pandas, SciPy, Matplotlib, PyYAML

---
*GSoC 2026 Project - Human AI*

# SIRA: Stochastic Infection Recovery Analysis

ML pipeline for learning the SIR epidemic vector field from stochastic simulations and recovering interpretable governing equations via sparse regression.

**Input**: (S, I, R) — normalised population fractions  
**Output**: (dS/dt, dI/dt, dR/dt) — instantaneous vector field

The true governing dynamics are:

```
dS/dt = -β·S·I
dI/dt =  β·S·I - γ·I
dR/dt =  γ·I
```

The pipeline learns this purely from noisy Gillespie simulation data without knowing the form in advance, then uses SINDy sparse regression to recover the symbolic equations.

## Architecture

```
Gillespie simulations (β,γ sampled uniformly)
  → ensemble average over multiple runs
  → finite-difference derivative estimation
  → VectorFieldDataset (parameter-stratified train/val/test split)
  → MLP training:  F(S,I,R) = (dS/dt, dI/dt, dR/dt)
  → SINDy sparse regression on the same dataset
  → FastAPI inference server
```

The parameter-stratified split matters: trajectories sharing (β,γ) values are kept in the same fold, so validation measures generalisation to unseen parameter regimes rather than interpolation within known ones.

## Design decisions

- **Finite-difference derivatives**: `np.gradient` over ensemble-averaged trajectories. Averaging 20+ runs before differentiating suppresses most Gillespie noise; no smoothing filter is applied deliberately, to keep the pipeline simple and the estimation transparent.
- **SINDy feature library**: degree-2 polynomial in {S, I, R} (10 terms). This matches the structure of the true SIR ODEs without encoding prior knowledge about which terms survive.
- **Conservation constraint**: `PhysicsInformedMLP` enforces dS+dI+dR=0 via mean-subtraction on the output. The plain `VectorFieldMLP` does not enforce this; it is learned implicitly from the data.
- **No normalisation by default**: state variables are already in [0,1]; the derivative magnitudes are small but consistent across parameter settings, so standardisation is optional.

## Project structure

```
config/          YAML experiment configurations
data/            Raw and processed data (gitignored)
models/          Saved model checkpoints (gitignored)
notebooks/       Exploratory Jupyter notebooks
results/         Generated plots (gitignored)
scripts/         Experiment runner, hyperparameter sweep, deployment
src/
  core/          SIR simulator (Gillespie + deterministic ODE)
  data/          Dataset generation, splitting, preprocessing, validation
  evaluation/    Benchmarking and OOD detection
  inference/     Model predictor and FastAPI server
  models/        MLP architectures and model registry
  symbolic/      SINDy, PySR wrapper, expression utilities
  training/      Trainer, losses, metrics, callbacks
  visualization/ Parity and trajectory plots
tests/
  unit/          Component-level tests
  integration/   End-to-end pipeline tests
  regression/    Model performance threshold tests
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.9+. PyTorch CPU-only is sufficient for the default configuration.

## Quick start

```bash
# 1. Generate dataset (~500 parameter points × 20 runs each)
python src/generate_data.py

# 2. Train MLP and run SINDy baseline
python src/train_ml.py

# 3. Parity plots
python src/visualize_results.py
```

All three scripts accept `--help` for available options.

## Full experiment run

```bash
# Single command: data generation + training + symbolic regression
python scripts/run_experiment.py --config config/base.yaml

# Fast demo run (small dataset, 1 epoch — useful for CI validation)
python scripts/run_experiment.py \
      --config config/submission_base.yaml \
      --data-config config/submission_data.yaml \
      --model-config config/submission_model.yaml

# Hyperparameter sweep
python scripts/sweep_hyperparams.py

# Start inference server (requires a trained model at models/vector_field_mlp.pth)
python scripts/deploy_model.py --model-path models/vector_field_mlp.pth
```

## Inference API

```bash
# Health check
curl http://localhost:8000/health

# Predict vector field at a state point
curl -X POST http://localhost:8000/predict/vector_field \
     -H "Content-Type: application/json" \
     -d '{"S": 0.9, "I": 0.05, "R": 0.05}'

# Euler-integrate a trajectory
curl -X POST http://localhost:8000/simulate/trajectory \
     -H "Content-Type: application/json" \
     -d '{"S0": 0.99, "I0": 0.01, "R0": 0.0, "num_steps": 100, "dt": 0.5}'
```

## Testing

```bash
# Unit and API tests (~20 tests, runs in <10 s)
python -m pytest tests/unit/ -v

# Full suite including integration and regression
python -m pytest tests/ -v

# End-to-end smoke run (generates data, trains, plots)
python tests/smoke_test.py --num-param-points 8 --num-runs-per-param 3 --epochs 1
```

## Model architectures

| Model | Notes |
|-------|-------|
| `VectorFieldMLP` | 3-layer MLP with Tanh, no physics constraint (default) |
| `PhysicsInformedMLP` | Same architecture, output projected to satisfy dS+dI+dR=0 |
| `NeuralODEBaseline` | Euler-step integrator; takes (S,I,R,dt) as input |

## Technology

Python 3.9+, PyTorch, NumPy, SciPy, FastAPI, PyYAML, MLflow (optional).


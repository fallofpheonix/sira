# SIRA

SIRA trains a neural approximation of the SIR vector field from stochastic epidemic simulations, then exposes the trained model behind a small FastAPI service.

The target dynamics are the standard SIR system:

```text
dS/dt = -β·S·I
dI/dt =  β·S·I - γ·I
dR/dt =  γ·I
```

## What It Does

- generates ensemble-averaged SIR trajectories from Gillespie simulations,
- trains a vector-field model on $(S, I, R) \rightarrow (dS/dt, dI/dt, dR/dt)$,
- emits a parity plot for a quick sanity check,
- serves prediction and trajectory endpoints for downstream consumers.

## Architecture

```text
Gillespie simulations
  -> ensemble averaging
  -> finite-difference derivative estimation
  -> vector-field training
  -> SINDy sparse regression
  -> FastAPI inference service
```

## Layout

```text
src/sira/
  api/        HTTP entrypoints and request schemas
  config/     YAML loading and project-level settings helpers
  core/       path and bootstrap primitives shared across layers
  services/   dataset, training, reporting, and experiment orchestration
  utils/      small cross-cutting helpers
```

Legacy domain modules under `src/` are still present because they hold the underlying simulator, model definitions, and symbolic tooling. The newer `sira` package is the boundary used by scripts and tests.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .

python src/generate_data.py --num-param-points 50 --num-runs-per-param 5
python src/train_ml.py --epochs 5
python src/visualize_results.py
python scripts/deploy_model.py --model-path models/vector_field_mlp.pth
```

For the end-to-end workflow:

```bash
python scripts/run_experiment.py --config config/base.yaml
```

## Key Decisions

- The service layer is intentionally thin. The older domain modules still do the math-heavy work; orchestration lives in `src/sira/services`.
- The API uses an app factory and explicit service state instead of module-level globals. It is easier to test and less fragile during startup.
- Dependency choices stay narrow on purpose. There is no separate task runner or config framework yet because the repo is still small.

## Test

```bash
python -m pytest tests/unit/test_api.py tests/integration/test_pipeline.py -q
python tests/smoke_test.py --num-param-points 8 --num-runs-per-param 3 --num-timepoints 20 --epochs 1
```

# SIRA Submission Package

## Project Overview

SIRA builds a simulation-informed ML pipeline for SIR epidemics:

1. Generate stochastic trajectories with Gillespie simulation.
2. Learn the vector field $(S, I, R) \rightarrow (dS/dt, dI/dt, dR/dt)$.
3. Recover symbolic dynamics using SINDy.
4. Expose model inference through a FastAPI API.

This `submission/` folder is self-contained for review and execution.

## Folder Contents

- `source_code/` production-ready runnable code snapshot.
- `ARCHITECTURE.md` design and module responsibilities.
- `API_DOCUMENTATION.md` API contract and usage.
- `DEMO_INSTRUCTIONS.md` reviewer demo path.
- `TEST_VALIDATION.md` executed test evidence.
- `OUTPUT_SAMPLES.md` sample artifacts.
- `output_samples/` generated output examples.

## Setup Instructions

```bash
cd source_code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run Instructions

```bash
cd source_code
source .venv/bin/activate

# Full pipeline
python scripts/run_experiment.py --config config/base.yaml --data-config config/data.yaml --model-config config/model.yaml

# Start API server
python scripts/deploy_model.py --model-path models/vector_field_mlp.pth
```

## Dependencies

Dependencies are pinned in `source_code/requirements.txt` and package metadata in `source_code/pyproject.toml`.
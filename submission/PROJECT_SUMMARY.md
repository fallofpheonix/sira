# Project Summary

## Scope

SIRA implements a reproducible machine learning pipeline for recovering the SIR epidemic vector field from stochastic simulations and approximating the governing equations with symbolic regression.

## Requested Deliverables

1. Simulation Model
   Implemented through the stochastic Gillespie SIR simulator and data-generation pipeline in `src/core/` and `src/data/`.
2. Predictive ML Model
   Implemented through the vector-field MLP training pipeline in `src/models/`, `src/training/`, and `src/train_ml.py`.
3. Symbolic Approximation
   Implemented through the SINDy-based symbolic recovery step in `src/symbolic/` and `scripts/run_experiment.py`.

## Verified Execution Paths

- Targeted tests: `python -m pytest -q tests/smoke_test.py tests/unit tests/integration/test_pipeline.py`
- Fast smoke run: `python tests/smoke_test.py --num-param-points 8 --num-runs-per-param 3 --num-timepoints 20 --epochs 1`
- End-to-end demo run: `python scripts/run_experiment.py --config config/submission_base.yaml --data-config config/submission_data.yaml --model-config config/submission_model.yaml`

## Primary Repository Artifacts

- Dataset output: `data/processed/*.csv`
- Trained model: `models/vector_field_mlp.pth`
- Symbolic equations: `results/submission_demo/symbolic_expression.txt`
- Visual diagnostics: `results/vector_field_parity.png`
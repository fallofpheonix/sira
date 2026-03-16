# Run Instructions

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Validate the Repository

```bash
python -m pytest -q tests/smoke_test.py tests/unit tests/integration/test_pipeline.py
python tests/smoke_test.py --num-param-points 8 --num-runs-per-param 3 --num-timepoints 20 --epochs 1
```

## Fast End-to-End Demo

```bash
python scripts/run_experiment.py \
  --config config/submission_base.yaml \
  --data-config config/submission_data.yaml \
  --model-config config/submission_model.yaml
```

This demo run generates a small dataset, trains the vector-field model, and writes symbolic regression output to `results/submission_demo/`.

## Optional API Launch

```bash
python scripts/deploy_model.py --model-path models/vector_field_mlp.pth
```
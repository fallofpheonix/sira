# Demo Instructions / Usage Guide

## Quick Reviewer Demo

```bash
cd source_code
source .venv/bin/activate
python scripts/run_experiment.py \
  --config config/submission_base.yaml \
  --data-config config/submission_data.yaml \
  --model-config config/submission_model.yaml
```

Expected outputs:
- `source_code/data/processed/submission_demo_vector_field.csv`
- `source_code/models/vector_field_mlp.pth`
- `source_code/results/submission_demo/symbolic_expression.txt`

## API Demo

In one terminal:
```bash
cd source_code
source .venv/bin/activate
python scripts/deploy_model.py --model-path models/vector_field_mlp.pth
```

In another terminal:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict/vector_field \
  -H "Content-Type: application/json" \
  -d '{"S": 0.9, "I": 0.05, "R": 0.05}'
```

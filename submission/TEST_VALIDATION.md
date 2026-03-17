# Test Cases / Validation Proof

Validation was executed on macOS with Python 3.13 virtual environment.

## Automated Tests

```bash
cd source_code
source .venv/bin/activate
python -m pytest tests/unit/test_api.py tests/unit/test_services.py tests/integration/test_pipeline.py -q
```

Expected: all tests pass.

## End-to-End Smoke Validation

```bash
cd source_code
source .venv/bin/activate
python tests/smoke_test.py --num-param-points 8 --num-runs-per-param 3 --num-timepoints 20 --epochs 1
```

Expected:
- dataset generated,
- model saved,
- symbolic expressions printed,
- parity plot saved.

## Full Pipeline Validation

```bash
cd source_code
source .venv/bin/activate
python scripts/run_experiment.py --config config/base.yaml --data-config config/data.yaml --model-config config/model.yaml
```

Expected: command completes with dataset/model/symbolic output paths.

## Live API Validation

```bash
cd source_code
source .venv/bin/activate
python scripts/deploy_model.py --model-path models/vector_field_mlp.pth --host 127.0.0.1 --port 8010

# Separate terminal
curl http://127.0.0.1:8010/health
curl -X POST http://127.0.0.1:8010/predict/vector_field \
	-H "Content-Type: application/json" \
	-d '{"S":0.9,"I":0.05,"R":0.05}'
```

Expected:
- health response returns `{"status":"ok","model_loaded":true}`,
- prediction response returns numeric `dS_dt`, `dI_dt`, `dR_dt`.

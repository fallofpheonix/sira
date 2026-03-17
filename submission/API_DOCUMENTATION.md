# API Documentation

Base URL: `http://localhost:8000`

## `GET /health`

Returns service health and model availability.

Response example:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

## `GET /models`

Returns registered model names.

Response example:
```json
{
  "models": ["VectorFieldMLP", "NeuralODEBaseline", "PhysicsInformedMLP"]
}
```

## `POST /predict/vector_field`

Predicts derivatives at a state.

Request body:
```json
{
  "S": 0.9,
  "I": 0.05,
  "R": 0.05
}
```

Response body:
```json
{
  "dS_dt": -0.02,
  "dI_dt": 0.01,
  "dR_dt": 0.01
}
```

## `POST /simulate/trajectory`

Generates an Euler-integrated trajectory from the learned vector field.

Request body:
```json
{
  "S0": 0.99,
  "I0": 0.01,
  "R0": 0.0,
  "num_steps": 100,
  "dt": 0.1
}
```

Response includes arrays: `t`, `S`, `I`, `R`.

## Launch API

```bash
cd source_code
source .venv/bin/activate
python scripts/deploy_model.py --model-path models/vector_field_mlp.pth
```

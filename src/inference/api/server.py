from fastapi import FastAPI, HTTPException
from pathlib import Path
import os

app = FastAPI(title="SIRA Inference API")


@app.on_event("startup")
async def startup():
    model_path = os.environ.get("MODEL_PATH", "models/vector_field_mlp.pth")
    app.state.model_path = model_path
    if Path(model_path).exists():
        from src.inference.predictor import VectorFieldPredictor
        predictor = VectorFieldPredictor(model_path)
        predictor.load_model()
        app.state.predictor = predictor
    else:
        app.state.predictor = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.predictor is not None}


@app.get("/models")
def list_models():
    from src.models.registry import list_models
    return {"models": list_models()}


@app.post("/predict/vector_field")
def predict_vector_field(request: "VectorFieldRequest"):
    from src.inference.api.schemas import VectorFieldRequest, VectorFieldResponse
    if app.state.predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    dS, dI, dR = app.state.predictor.predict(request.S, request.I, request.R)
    return VectorFieldResponse(dS_dt=dS, dI_dt=dI, dR_dt=dR)


@app.post("/simulate/trajectory")
def simulate_trajectory(request: "TrajectoryRequest"):
    from src.inference.api.schemas import TrajectoryRequest, TrajectoryResponse
    if app.state.predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    traj = app.state.predictor.predict_trajectory(
        request.S0, request.I0, request.R0, request.num_steps, request.dt
    )
    return TrajectoryResponse(**traj)


@app.get("/models/{model_id}/equation")
def get_equation(model_id: str):
    return {"model_id": model_id, "equation": "Not available for neural models"}

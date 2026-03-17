from fastapi import APIRouter, HTTPException, Request

from sira.api.schemas import (
    TrajectoryRequest,
    TrajectoryResponse,
    VectorFieldRequest,
    VectorFieldResponse,
)
from sira.services.model_catalog import list_available_models


router = APIRouter()


def _service_from(request: Request):
    return request.app.state.inference_service


@router.get("/health")
def healthcheck(request: Request) -> dict[str, bool | str]:
    service = _service_from(request)
    return {"status": "ok", "model_loaded": service.is_ready}


@router.get("/models")
def list_models() -> dict[str, list[str]]:
    return {"models": list_available_models()}


@router.post("/predict/vector_field", response_model=VectorFieldResponse)
def predict_vector_field(request_body: VectorFieldRequest, request: Request) -> VectorFieldResponse:
    service = _service_from(request)
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    d_s, d_i, d_r = service.predict_vector_field(request_body.S, request_body.I, request_body.R)
    return VectorFieldResponse(dS_dt=d_s, dI_dt=d_i, dR_dt=d_r)


@router.post("/simulate/trajectory", response_model=TrajectoryResponse)
def simulate_trajectory(request_body: TrajectoryRequest, request: Request) -> TrajectoryResponse:
    service = _service_from(request)
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    trajectory = service.simulate_trajectory(
        request_body.S0,
        request_body.I0,
        request_body.R0,
        request_body.num_steps,
        request_body.dt,
    )
    return TrajectoryResponse(**trajectory)


@router.get("/models/{model_id}/equation")
def get_equation(model_id: str) -> dict[str, str]:
    return {"model_id": model_id, "equation": "Not available for neural models"}
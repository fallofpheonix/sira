from sira.api.schemas import (
    TrajectoryRequest,
    TrajectoryResponse,
    VectorFieldRequest,
    VectorFieldResponse,
)
from pydantic import BaseModel


class ModelInfo(BaseModel):
    model_id: str
    name: str
    input_dim: int
    output_dim: int

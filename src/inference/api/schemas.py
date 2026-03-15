from pydantic import BaseModel


class VectorFieldRequest(BaseModel):
    S: float
    I: float
    R: float


class VectorFieldResponse(BaseModel):
    dS_dt: float
    dI_dt: float
    dR_dt: float


class TrajectoryRequest(BaseModel):
    S0: float
    I0: float
    R0: float
    num_steps: int = 100
    dt: float = 0.1


class TrajectoryResponse(BaseModel):
    t: list
    S: list
    I: list
    R: list


class ModelInfo(BaseModel):
    model_id: str
    name: str
    input_dim: int
    output_dim: int

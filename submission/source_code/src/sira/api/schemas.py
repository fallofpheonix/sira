from pydantic import BaseModel, Field


class VectorFieldRequest(BaseModel):
    S: float = Field(ge=0.0, le=1.0)
    I: float = Field(ge=0.0, le=1.0)
    R: float = Field(ge=0.0, le=1.0)


class VectorFieldResponse(BaseModel):
    dS_dt: float
    dI_dt: float
    dR_dt: float


class TrajectoryRequest(BaseModel):
    S0: float = Field(ge=0.0, le=1.0)
    I0: float = Field(ge=0.0, le=1.0)
    R0: float = Field(ge=0.0, le=1.0)
    num_steps: int = Field(default=100, ge=1, le=2000)
    dt: float = Field(default=0.1, gt=0.0, le=5.0)


class TrajectoryResponse(BaseModel):
    t: list[float]
    S: list[float]
    I: list[float]
    R: list[float]
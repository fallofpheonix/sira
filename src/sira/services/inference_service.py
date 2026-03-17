from dataclasses import dataclass
from pathlib import Path

from src.inference.predictor import VectorFieldPredictor


@dataclass(slots=True)
class InferenceService:
    model_path: Path
    predictor: VectorFieldPredictor | None = None

    @classmethod
    def from_model_path(cls, model_path: str | Path) -> "InferenceService":
        path = Path(model_path)
        service = cls(model_path=path)
        if path.exists():
            predictor = VectorFieldPredictor(path)
            predictor.load_model()
            service.predictor = predictor
        return service

    @property
    def is_ready(self) -> bool:
        return self.predictor is not None

    def predict_vector_field(self, susceptible: float, infected: float, recovered: float) -> tuple[float, float, float]:
        if self.predictor is None:
            raise RuntimeError("Model not loaded")
        return self.predictor.predict(susceptible, infected, recovered)

    def simulate_trajectory(
        self,
        susceptible: float,
        infected: float,
        recovered: float,
        num_steps: int,
        dt: float,
    ) -> dict[str, list[float]]:
        if self.predictor is None:
            raise RuntimeError("Model not loaded")
        return self.predictor.predict_trajectory(susceptible, infected, recovered, num_steps, dt)
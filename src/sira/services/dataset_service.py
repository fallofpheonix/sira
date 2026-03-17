from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.generator import DataPipeline


@dataclass(slots=True)
class DatasetBuildRequest:
    output_path: Path
    num_param_points: int = 500
    num_runs_per_param: int = 20
    population: int = 1000
    num_timepoints: int = 100
    beta_min: float = 0.1
    beta_max: float = 0.9
    gamma_min: float = 0.02
    gamma_max: float = 0.4
    seed: int = 42
    max_time: float = 150.0


class DatasetService:
    def build_vector_field_dataset(self, request: DatasetBuildRequest) -> pd.DataFrame:
        if request.num_param_points < 1 or request.num_runs_per_param < 1:
            raise ValueError("Dataset generation requires at least one parameter point and one run")
        if request.population < 10:
            raise ValueError("Population is too small for a stable demo dataset")

        pipeline = DataPipeline(
            {
                "population": request.population,
                "beta_min": request.beta_min,
                "beta_max": request.beta_max,
                "gamma_min": request.gamma_min,
                "gamma_max": request.gamma_max,
                "seed": request.seed,
                "num_param_points": request.num_param_points,
                "num_runs_per_param": request.num_runs_per_param,
                "num_timepoints": request.num_timepoints,
                "max_time": request.max_time,
            }
        )
        return pipeline.run(
            output_path=request.output_path,
            num_param_points=request.num_param_points,
            num_runs_per_param=request.num_runs_per_param,
            num_timepoints=request.num_timepoints,
            max_time=request.max_time,
        )
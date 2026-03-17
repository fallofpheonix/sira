from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from src.models.registry import get_model
from src.visualization.plots import plot_parity
from sira.utils.randomness import seed_everything


@dataclass(slots=True)
class ReportBuildRequest:
    dataset_path: Path
    model_path: Path
    output_path: Path
    sample_size: int = 5000
    seed: int = 0
    model_name: str = "VectorFieldMLP"
    hidden_dim: int = 128
    num_layers: int = 3
    activation: str = "tanh"
    dropout: float = 0.0


class ReportingService:
    def build_parity_report(self, request: ReportBuildRequest) -> Path:
        if not request.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {request.dataset_path}")
        if not request.model_path.exists():
            raise FileNotFoundError(f"Model not found at {request.model_path}")

        seed_everything(request.seed)
        frame = pd.read_csv(request.dataset_path)
        if frame.empty:
            raise ValueError("Dataset is empty")

        sample_size = min(request.sample_size, len(frame))
        sample = frame.sample(n=sample_size, random_state=request.seed)

        model = get_model(
            request.model_name,
            hidden_dim=request.hidden_dim,
            num_layers=request.num_layers,
            activation=request.activation,
            dropout=request.dropout,
        )
        model.load_state_dict(torch.load(request.model_path, map_location="cpu"))
        model.eval()

        features = torch.tensor(sample[["S", "I", "R"]].values, dtype=torch.float32)
        with torch.no_grad():
            predictions = model(features).numpy()

        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_parity(
            sample[["dS_dt", "dI_dt", "dR_dt"]].values,
            predictions,
            output_path=request.output_path,
        )
        return request.output_path
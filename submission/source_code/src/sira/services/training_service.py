from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import DatasetSplitter, VectorFieldDataset
from src.models.registry import get_model
from src.training.trainer import Trainer
from sira.utils.randomness import seed_everything


@dataclass(slots=True)
class TrainingRunRequest:
    dataset_path: Path
    model_output_path: Path
    model_name: str = "VectorFieldMLP"
    hidden_dim: int = 128
    num_layers: int = 3
    activation: str = "tanh"
    dropout: float = 0.0
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 100
    seed: int = 42
    checkpoint_dir: Path | None = None


@dataclass(slots=True)
class TrainingRunResult:
    rows_seen: int
    model_path: Path
    checkpoint_path: Path
    history: dict[str, list[float]]


class TrainingService:
    def train_vector_field_model(self, request: TrainingRunRequest) -> TrainingRunResult:
        if not request.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {request.dataset_path}")

        seed_everything(request.seed)
        frame = pd.read_csv(request.dataset_path)
        if frame.empty:
            raise ValueError("Dataset is empty")

        dataset = VectorFieldDataset(frame)
        splitter = DatasetSplitter()
        train_split, validation_split, _ = splitter.split(dataset, seed=request.seed)

        train_loader = DataLoader(train_split, batch_size=request.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_split, batch_size=request.batch_size, shuffle=False)

        model_kwargs = {
            "hidden_dim": request.hidden_dim,
            "num_layers": request.num_layers,
            "activation": request.activation,
            "dropout": request.dropout,
        }
        model = get_model(request.model_name, **model_kwargs)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=request.learning_rate,
            weight_decay=request.weight_decay,
        )
        trainer = Trainer(
            model,
            optimizer,
            torch.nn.MSELoss(),
            config={"checkpoint_dir": str(request.checkpoint_dir or request.model_output_path.parent / "checkpoints")},
            model_kwargs=model_kwargs,
        )
        history = trainer.fit(train_loader, validation_loader, epochs=request.epochs)

        request.model_output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = request.model_output_path.with_name(f"{request.model_output_path.stem}.ckpt")
        trainer.save_checkpoint(checkpoint_path, request.epochs, history["val_loss"][-1] if history["val_loss"] else 0.0)
        torch.save(model.state_dict(), request.model_output_path)

        return TrainingRunResult(
            rows_seen=len(frame),
            model_path=request.model_output_path,
            checkpoint_path=checkpoint_path,
            history=history,
        )
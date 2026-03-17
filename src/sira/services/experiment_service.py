from dataclasses import dataclass
from pathlib import Path

import torch

from src.symbolic.sindy import SINDy
from sira.config.loader import load_yaml_file
from sira.services.dataset_service import DatasetBuildRequest, DatasetService
from sira.services.training_service import TrainingRunRequest, TrainingService
from sira.utils.randomness import seed_everything


@dataclass(slots=True)
class ExperimentResult:
    dataset_path: Path
    model_path: Path
    symbolic_expression_path: Path


class ExperimentService:
    def __init__(self) -> None:
        self.dataset_service = DatasetService()
        self.training_service = TrainingService()

    def run(self, base_config_path: str | Path, data_config_path: str | Path, model_config_path: str | Path) -> ExperimentResult:
        base_config = load_yaml_file(base_config_path)
        data_config = load_yaml_file(data_config_path).get("data", {})
        model_config = load_yaml_file(model_config_path)

        experiment_config = base_config.get("experiment", {})
        seed = experiment_config.get("seed", 42)
        seed_everything(seed)

        dataset_path = Path(data_config.get("output_path", "data/processed/sir_vector_field.csv"))
        dataset = self.dataset_service.build_vector_field_dataset(
            DatasetBuildRequest(
                output_path=dataset_path,
                num_param_points=data_config.get("num_param_points", 50),
                num_runs_per_param=data_config.get("num_runs_per_param", 5),
                population=data_config.get("population", 1000),
                num_timepoints=data_config.get("num_timepoints", 50),
                beta_min=data_config.get("beta_min", 0.1),
                beta_max=data_config.get("beta_max", 0.9),
                gamma_min=data_config.get("gamma_min", 0.02),
                gamma_max=data_config.get("gamma_max", 0.4),
                seed=seed,
                max_time=data_config.get("max_time", 150),
            )
        )

        model_section = model_config.get("model", {})
        training_section = model_config.get("training", {})
        model_path = Path(training_section.get("output_path", "models/vector_field_mlp.pth"))
        self.training_service.train_vector_field_model(
            TrainingRunRequest(
                dataset_path=dataset_path,
                model_output_path=model_path,
                model_name=model_section.get("name", "VectorFieldMLP"),
                hidden_dim=model_section.get("hidden_dim", 128),
                num_layers=model_section.get("num_layers", 3),
                activation=model_section.get("activation", "tanh"),
                dropout=model_section.get("dropout", 0.0),
                batch_size=training_section.get("batch_size", 256),
                learning_rate=training_section.get("learning_rate", 1e-3),
                weight_decay=training_section.get("weight_decay", 0.0),
                epochs=training_section.get("epochs", 100),
                seed=seed,
                checkpoint_dir=Path(training_section.get("checkpoint_dir", "checkpoints")),
            )
        )

        output_dir = Path(experiment_config.get("output_dir", "results/default_run"))
        output_dir.mkdir(parents=True, exist_ok=True)
        expression_path = output_dir / "symbolic_expression.txt"

        sindy = SINDy()
        sindy.fit(dataset)
        sindy.save(expression_path)

        # TODO: persist metrics once the training loop exposes a stable summary payload.
        return ExperimentResult(dataset_path=dataset_path, model_path=model_path, symbolic_expression_path=expression_path)
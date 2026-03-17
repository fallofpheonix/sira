from pathlib import Path

import pandas as pd
import pytest

from sira.services.dataset_service import DatasetBuildRequest, DatasetService
from sira.services.training_service import TrainingRunRequest, TrainingService


def test_dataset_service_rejects_tiny_population(tmp_path):
    service = DatasetService()
    request = DatasetBuildRequest(
        output_path=tmp_path / "dataset.csv",
        num_param_points=2,
        num_runs_per_param=1,
        population=5,
    )
    with pytest.raises(ValueError):
        service.build_vector_field_dataset(request)


def test_training_service_rejects_empty_dataset(tmp_path):
    csv_path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["S", "I", "R", "dS_dt", "dI_dt", "dR_dt"]).to_csv(csv_path, index=False)

    service = TrainingService()
    request = TrainingRunRequest(dataset_path=csv_path, model_output_path=tmp_path / "model.pth", epochs=1)

    with pytest.raises(ValueError):
        service.train_vector_field_model(request)
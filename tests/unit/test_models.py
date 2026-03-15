import torch
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.architectures.mlp import VectorFieldMLP
from src.models.architectures.physics_informed import PhysicsInformedMLP
from src.models.registry import get_model, list_models


def test_mlp_forward():
    model = VectorFieldMLP()
    x = torch.randn(16, 3)
    out = model(x)
    assert out.shape == (16, 3)


def test_physics_informed_conservation():
    model = PhysicsInformedMLP()
    x = torch.randn(32, 3)
    out = model(x)
    assert out.shape == (32, 3)
    # Sum of outputs should be ~0 due to conservation constraint
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.zeros(32), atol=1e-5)


def test_model_save_load(tmp_path):
    model = VectorFieldMLP()
    path = tmp_path / "test_model.pth"
    torch.save(model.state_dict(), path)
    model2 = VectorFieldMLP()
    model2.load_state_dict(torch.load(path, map_location='cpu'))
    x = torch.randn(4, 3)
    assert torch.allclose(model(x), model2(x))


def test_registry():
    names = list_models()
    assert 'VectorFieldMLP' in names
    model = get_model('VectorFieldMLP', hidden_dim=64)
    assert isinstance(model, VectorFieldMLP)

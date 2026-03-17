"""API smoke tests for the SIRA FastAPI inference server.

Uses FastAPI's TestClient (backed by httpx) so no real server process is needed.
The tests cover two scenarios:
  - no model loaded (predictor is None): health, /models, /predict → 503
  - model loaded in-memory: /health, /predict/vector_field, /simulate/trajectory
"""
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
import pytest
import torch

from src.inference.api.server import app
from src.models.architectures.mlp import VectorFieldMLP
from src.inference.predictor import VectorFieldPredictor
from sira.services.inference_service import InferenceService


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_predictor(tmp_path, hidden_dim=32):
    """Create and save a tiny model, then return a loaded predictor."""
    model = VectorFieldMLP(hidden_dim=hidden_dim)
    model_path = tmp_path / "test_model.pth"
    torch.save(model.state_dict(), model_path)
    predictor = VectorFieldPredictor(
        model_path=str(model_path),
        model_kwargs={"hidden_dim": hidden_dim},
    )
    predictor.load_model()
    return predictor


def _make_service(tmp_path, hidden_dim=32):
    service = InferenceService(model_path=Path(tmp_path) / "test_model.pth")
    service.predictor = _make_predictor(tmp_path, hidden_dim=hidden_dim)
    return service


# ── Tests: no model loaded ────────────────────────────────────────────────────

class TestNoModel:
    """Endpoints that operate without a loaded model."""

    def setup_method(self):
        app.state.inference_service = InferenceService(model_path=Path("missing-model.pth"))
        self.client = TestClient(app, raise_server_exceptions=True)

    def test_health_no_model(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is False

    def test_list_models(self):
        r = self.client.get("/models")
        assert r.status_code == 200
        names = r.json()["models"]
        assert "VectorFieldMLP" in names

    def test_predict_without_model_returns_503(self):
        r = self.client.post(
            "/predict/vector_field",
            json={"S": 0.9, "I": 0.05, "R": 0.05},
        )
        assert r.status_code == 503

    def test_trajectory_without_model_returns_503(self):
        r = self.client.post(
            "/simulate/trajectory",
            json={"S0": 0.99, "I0": 0.01, "R0": 0.0, "num_steps": 5, "dt": 0.1},
        )
        assert r.status_code == 503

    def test_equation_endpoint(self):
        r = self.client.get("/models/VectorFieldMLP/equation")
        assert r.status_code == 200
        assert "model_id" in r.json()


# ── Tests: model loaded ───────────────────────────────────────────────────────

class TestWithModel:
    """Endpoints that require a loaded predictor."""

    def setup_method(self, tmp_path_factory=None):
        self._tmpdir = Path(tempfile.mkdtemp())
        app.state.inference_service = _make_service(self._tmpdir)
        self.client = TestClient(app, raise_server_exceptions=True)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        app.state.inference_service = InferenceService(model_path=Path("missing-model.pth"))

    def test_health_with_model(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is True

    def test_predict_vector_field_shape(self):
        r = self.client.post(
            "/predict/vector_field",
            json={"S": 0.9, "I": 0.05, "R": 0.05},
        )
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) == {"dS_dt", "dI_dt", "dR_dt"}
        for v in body.values():
            assert isinstance(v, float)

    def test_predict_vector_field_conservation(self):
        """dS + dI + dR ≈ 0 is not guaranteed for plain MLP, but outputs must be finite."""
        r = self.client.post(
            "/predict/vector_field",
            json={"S": 0.8, "I": 0.15, "R": 0.05},
        )
        assert r.status_code == 200
        body = r.json()
        import math
        for v in body.values():
            assert math.isfinite(v)

    def test_simulate_trajectory_length(self):
        num_steps = 10
        r = self.client.post(
            "/simulate/trajectory",
            json={"S0": 0.99, "I0": 0.01, "R0": 0.0, "num_steps": num_steps, "dt": 0.5},
        )
        assert r.status_code == 200
        body = r.json()
        # trajectory contains t=0 plus num_steps steps
        assert len(body["t"]) == num_steps + 1
        assert len(body["S"]) == num_steps + 1
        assert len(body["I"]) == num_steps + 1
        assert len(body["R"]) == num_steps + 1

    def test_simulate_trajectory_initial_state(self):
        r = self.client.post(
            "/simulate/trajectory",
            json={"S0": 0.99, "I0": 0.01, "R0": 0.0, "num_steps": 5, "dt": 0.1},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["t"][0] == pytest.approx(0.0)
        assert body["S"][0] == pytest.approx(0.99)
        assert body["I"][0] == pytest.approx(0.01)
        assert body["R"][0] == pytest.approx(0.0)

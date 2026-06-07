import torch
import numpy as np
from pathlib import Path
import re
from src.models.registry import get_model


class VectorFieldPredictor:
    def __init__(self, model_path, device='cpu', model_name=None, model_kwargs=None):
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self.model = None

    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Checkpoint produced by Trainer.save_checkpoint(); may carry metadata.
            name = self.model_name or checkpoint.get('model_name', 'VectorFieldMLP')
            kwargs = self.model_kwargs or checkpoint.get('model_kwargs', {})
            try:
                self.model = get_model(name, **kwargs)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"Failed to reconstruct model '{name}' from checkpoint "
                    f"'{self.model_path}': {exc}"
                ) from exc
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Bare state_dict (e.g. torch.save(model.state_dict(), path)).
            name = self.model_name or 'VectorFieldMLP'
            inferred_kwargs = self._infer_model_kwargs_from_state_dict(checkpoint)
            effective_kwargs = {**inferred_kwargs, **self.model_kwargs}
            try:
                self.model = get_model(name, **effective_kwargs)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"Failed to instantiate model '{name}' for checkpoint "
                    f"'{self.model_path}': {exc}"
                ) from exc
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        return self

    @staticmethod
    def _infer_model_kwargs_from_state_dict(state_dict):
        if not isinstance(state_dict, dict):
            return {}

        first_layer = state_dict.get('net.0.weight')
        if first_layer is None or not hasattr(first_layer, 'shape') or len(first_layer.shape) != 2:
            return {}

        hidden_dim = int(first_layer.shape[0])

        linear_weight_pattern = re.compile(r"^net\.(\d+)\.weight$")
        linear_indices = []
        for key, tensor in state_dict.items():
            match = linear_weight_pattern.match(key)
            if not match or not hasattr(tensor, 'shape') or len(tensor.shape) != 2:
                continue
            linear_indices.append(int(match.group(1)))

        if not linear_indices:
            return {"hidden_dim": hidden_dim}

        linear_indices.sort()
        num_layers = max(len(linear_indices) - 1, 1)
        return {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "activation": "tanh",
            "dropout": 0.0,
        }

    def predict(self, S, I, R):
        if self.model is None:
            self.load_model()
        x = torch.tensor([[S, I, R]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(x).squeeze()
        return out[0].item(), out[1].item(), out[2].item()

    def predict_trajectory(self, S0, I0, R0, num_steps, dt=0.1):
        if self.model is None:
            self.load_model()
        S, I, R = S0, I0, R0
        trajectory = {'t': [0.0], 'S': [S], 'I': [I], 'R': [R]}
        for step in range(num_steps):
            dS, dI, dR = self.predict(S, I, R)
            S = max(0.0, S + dS * dt)
            I = max(0.0, I + dI * dt)
            R = max(0.0, R + dR * dt)
            trajectory['t'].append((step + 1) * dt)
            trajectory['S'].append(S)
            trajectory['I'].append(I)
            trajectory['R'].append(R)
        return trajectory

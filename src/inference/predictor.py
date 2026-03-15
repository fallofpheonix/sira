import torch
import numpy as np
from pathlib import Path
from src.models.architectures.mlp import VectorFieldMLP


class VectorFieldPredictor:
    def __init__(self, model_path, device='cpu'):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None

    def load_model(self):
        self.model = VectorFieldMLP()
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        return self

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

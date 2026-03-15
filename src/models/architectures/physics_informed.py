import torch
import torch.nn as nn


class PhysicsInformedMLP(nn.Module):
    """VectorFieldMLP with conservation constraint: dS+dI+dR=0."""
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=128, num_layers=3,
                 activation='tanh', dropout=0.0):
        super().__init__()
        act_fn = nn.Tanh() if activation == 'tanh' else nn.ReLU()
        layers = [nn.Linear(input_dim, hidden_dim), act_fn]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        # Project to satisfy sum=0: subtract mean across output dims
        out = out - out.mean(dim=-1, keepdim=True)
        return out

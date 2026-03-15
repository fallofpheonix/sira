import torch
import torch.nn as nn


class VectorFieldMLP(nn.Module):
    """Learns F(S,I,R) = (dS/dt, dI/dt, dR/dt)."""
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
        return self.net(x)

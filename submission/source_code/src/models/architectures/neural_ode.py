import torch
import torch.nn as nn


class NeuralODEBaseline(nn.Module):
    """Simple RNN-like ODE integrator using Euler steps. Input: (S, I, R, dt)."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x):
        # x: (batch, 4) where last col is dt
        return self.net(x)

    def integrate(self, S0, I0, R0, num_steps, dt=0.1):
        device = next(self.parameters()).device
        state = torch.tensor([[S0, I0, R0]], dtype=torch.float32, device=device)
        trajectory = [state.squeeze().tolist()]
        for _ in range(num_steps):
            dt_t = torch.full((1, 1), dt, device=device)
            inp = torch.cat([state, dt_t], dim=-1)
            dstate = self.forward(inp)
            state = state + dt * dstate
            trajectory.append(state.squeeze().tolist())
        return trajectory

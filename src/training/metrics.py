import torch
import numpy as np

# Small constant to prevent division by zero in R² computation
_EPS = 1e-10


def compute_r2(pred, target):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + _EPS))


def compute_rmse(pred, target):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    return float(np.sqrt(((target - pred) ** 2).mean()))


def compute_mae(pred, target):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    return float(np.abs(target - pred).mean())

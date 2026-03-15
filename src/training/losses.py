import torch
import torch.nn.functional as F


def mse_loss(pred, target):
    return F.mse_loss(pred, target)


def physics_loss(pred, target, lambda_physics=0.1):
    data_loss = F.mse_loss(pred, target)
    conservation = pred.sum(dim=-1).pow(2).mean()
    return data_loss + lambda_physics * conservation

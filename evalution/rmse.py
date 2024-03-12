import torch


def rmse(preds, targets):
    return torch.sqrt(torch.mean((preds - targets) ** 2))

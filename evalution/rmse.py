import torch


def rmse(preds, targets):
    torch.sqrt(torch.mean((preds - targets) ** 2))
    return torch.sqrt(torch.mean((preds - targets) ** 2))

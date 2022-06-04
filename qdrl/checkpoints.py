import torch
from torch import nn


def save_checkpoint(epoch: int, path: str, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

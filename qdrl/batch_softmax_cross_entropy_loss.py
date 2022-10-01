from typing import Dict

import numpy as np
import torch
from torch import nn

from qdrl.batch_predictor import predict
from qdrl.loss_computer import LossComputer


def batch_cosine(a: torch.tensor, b: torch.tensor, eps: float = 1e-8) -> torch.tensor:
    '''
    https://stackoverflow.com/a/58144658/7073537
    '''
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class BatchSoftmaxCrossEntropyLossComputer(LossComputer):
    def __init__(self, batch_size: int, temperature: float):
        self.target = torch.tensor(np.eye(batch_size))
        self.loss = nn.CrossEntropyLoss(reduce=True, reduction="mean")
        self.temperature = temperature

    def compute(self,
                model: nn.Module,
                batch: Dict[str, torch.Tensor],
                device: torch.device) -> torch.tensor:
        anchor_out, positive_out = predict(model, batch, device)
        # compute similarity matrix:
        # [cos(a1,p1), cos(a1, p2) ... cos(a1, pn)]
        # ...
        # [cos(an,p1), cos(an, p2) ... cos(an, pn)]
        similarity_matrix = batch_cosine(anchor_out, positive_out) / self.temperature
        return self.loss(similarity_matrix, self.target.to(device))

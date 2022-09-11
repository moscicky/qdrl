from typing import Tuple

import numpy as np
import torch
from torch import nn

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
    def __init__(self, batch_size: int):
        self.target = torch.tensor(np.eye(batch_size))
        self.loss = nn.CrossEntropyLoss(reduce=True, reduction="mean")

    def compute(self,
                model: nn.Module,
                batch: Tuple[torch.tensor, ...],
                device: torch.device) -> torch.tensor:
        anchor, positive = batch[0].to(device), batch[1].to(device)
        anchor_out = model(anchor)
        positive_out = model(positive)
        # compute similarity matrix:
        # [cos(a1,p1), cos(a1, p2) ... cos(a1, pn)]
        # ...
        # [cos(an,p1), cos(a1, p2) ... cos(an, pn)]
        similarity_matrix = batch_cosine(anchor_out, positive_out)
        return self.loss(similarity_matrix, self.target)

import abc
from typing import Tuple

import torch
from torch import nn


class LossComputer(abc.ABC):
    @abc.abstractmethod
    def compute(self,
                model: nn.Module,
                batch: Tuple[torch.tensor, ...],
                device: torch.device) -> torch.tensor:
        ...

import abc
from typing import Tuple, List

import torch
from torch import nn
from torch.cpu.amp import autocast


class TripletAssembler(abc.ABC):

    @abc.abstractmethod
    def generate_triplets(self,
                          model: nn.Module,
                          batch: Tuple[torch.tensor, ...],
                          device: torch.device
                          ) -> [torch.tensor,
                                torch.tensor,
                                torch.tensor]:
        pass


class BatchNegativeTripletsAssembler(TripletAssembler):
    def __init__(self, batch_size: int, negatives_count: int):
        self.anchor_mask, self.positive_mask, self.negative_mask = self.batch_negative_triplets_mask(batch_size,
                                                                                                     negatives_count)

    def generate_triplets(self,
                          model: nn.Module,
                          batch: Tuple[torch.tensor, ...],
                          device: torch.device
                          ) -> [torch.tensor,
                                torch.tensor,
                                torch.tensor]:
        anchor, positive = batch[0].to(device), batch[1].to(device)
        with autocast():
            anchor_out = model(anchor)
            positive_out = model(positive)

        anchor = anchor_out[self.anchor_mask]
        positive = positive_out[self.positive_mask]
        negative = positive_out[self.negative_mask]

        return anchor, positive, negative

    @staticmethod
    def batch_negative_triplets_mask(
            batch_size: int,
            negatives_count: int) -> [List[int], List[int], List[int]]:
        anchor_mask = []
        positive_mask = []
        negative_mask = []
        for anchor_idx in range(0, batch_size):
            for positive_idx in range(0, batch_size):
                # select top min(negatives_count, batch_size - 1) positives as negatives
                if anchor_idx != positive_idx and positive_idx <= negatives_count and positive_idx < batch_size:
                    anchor_mask.append(anchor_idx)
                    positive_mask.append(anchor_idx)
                    negative_mask.append(positive_idx)
        return anchor_mask, positive_mask, negative_mask

from typing import List, Dict

import torch
from torch import nn
import torch.nn.functional as F

from qdrl.batch_predictor import predict
from qdrl.loss_computer import LossComputer

class BatchNegativeTripletsAssembler:
    def __init__(self, batch_size: int, negatives_count: int):
        self.anchor_mask, self.positive_mask, self.negative_mask = self.batch_negative_triplets_mask(batch_size,
                                                                                                     negatives_count)

    def generate_triplets(self,
                          model: nn.Module,
                          batch: Dict[str, torch.Tensor],
                          device: torch.device
                          ) -> [torch.tensor,
                                torch.tensor,
                                torch.tensor]:
        anchor_out, positive_out = predict(model, batch, device)

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


class BatchTripletLossComputer(LossComputer):
    def __init__(self, batch_size: int, negatives_count: int, loss_margin: float):
        self.triplet_assembler = BatchNegativeTripletsAssembler(batch_size=batch_size, negatives_count=negatives_count)
        self.loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
                                                     margin=loss_margin)

    def compute(self, model: nn.Module, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.tensor:
        anchor, positive, negative = self.triplet_assembler.generate_triplets(model, batch, device)
        loss = self.loss(anchor=anchor, positive=positive, negative=negative)
        return loss

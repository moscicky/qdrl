import torch
from torch import nn
from torch.utils.data import DataLoader

from qdrl.triplets import TripletAssembler


class LossValidator:
    def __init__(self, dataloader: DataLoader,
                 loss_fn: nn.TripletMarginWithDistanceLoss,
                 triplet_assembler: TripletAssembler,
                 device: torch.device
                 ):
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.triplet_assembler = triplet_assembler
        self.device = device

    def validate(self, model: nn.Module, epoch: int):
        model.eval()
        validation_loss = 0.0
        print(f"Starting validation after epoch: {epoch}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                anchor, positive, negative = self.triplet_assembler.generate_triplets(model, batch, self.device)
                loss = self.loss_fn(anchor=anchor, positive=positive, negative=negative)
                batch_loss = loss.item()
                validation_loss += batch_loss
                if batch_idx % 10_000 == 0:
                    print(f"processed {batch_idx} validation batches")
        average_loss = validation_loss / batch_idx
        print(f"Finished validation after epoch: {epoch}, total loss: {validation_loss}, average loss: {average_loss}")
        return validation_loss, average_loss

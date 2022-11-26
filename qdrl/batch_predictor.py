from typing import Dict

import torch
from torch import nn

from qdrl.models import SimpleTextEncoder, TwoTower, MultiModalTwoTower


def predict(model: nn.Module,
            batch: Dict[str, torch.Tensor],
            device: torch.device,
            ) -> [torch.tensor, torch.tensor]:
    if isinstance(model, SimpleTextEncoder):
        query_text = batch["query"][model.query_text_feature].to(device)
        document_text = batch["document"][model.document_text_feature].to(device)
        query = model(query_text)
        document = model(document_text)
    elif isinstance(model, TwoTower):
        query_text = batch["query"][model.query_text_feature].to(device)
        document_text = batch["document"][model.document_text_feature].to(device)
        query = model.forward_query(query_text)
        document = model.forward_document(document_text)
    elif isinstance(model, MultiModalTwoTower):
        query_text = batch["query"][model.query_text_feature].to(device)
        document_text = batch["document"][model.document_text_feature].to(device)
        document_categorical_feature = batch["document"][model.document_categorical_feature].to(device)
        query = model.forward_query(query_text)
        document = model.forward_document(text=document_text, category=document_categorical_feature)
    else:
        raise ValueError("Unknown model type!")
    return query, document

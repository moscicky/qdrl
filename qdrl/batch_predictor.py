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
        product_text = batch["product"][model.product_text_feature].to(device)
        query = model(query_text)
        product = model(product_text)
    elif isinstance(model, TwoTower):
        query_text = batch["query"][model.query_text_feature].to(device)
        product_text = batch["product"][model.product_text_feature].to(device)
        query = model.forward_query(query_text)
        product = model(product_text)
    elif isinstance(model, MultiModalTwoTower):
        query_text = batch["query"][model.query_text_feature].to(device)
        product_text = batch["product"][model.product_text_feature].to(device)
        product_categorical_feature = batch["product"][model.product_categorical_feature].to(device)
        query = model.forward_query(query_text)
        product = model.forward_product(text=product_text, category=product_categorical_feature)
    else:
        raise ValueError("Unknown model type!")
    return query, product

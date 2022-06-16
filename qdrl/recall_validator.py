import json
import os
from typing import Dict, List

import numpy as np
import faiss
import torch
from torch import nn

from qdrl.configs import SimilarityMetric, WordVectorizerConfig, ModelConfig, Item, Query, QueryEmbedded, EmbeddedItem
from qdrl.models import SimpleTextEncoder
from qdrl.preprocess import vectorize_word, clean_phrase


def create_faiss_index(
        dim: int,
        vectors: np.ndarray,
        similarity_metric: SimilarityMetric):
    index = faiss.IndexFlat(dim, faiss.METRIC_INNER_PRODUCT)
    if similarity_metric == SimilarityMetric.COSINE:
        faiss.normalize_L2(vectors)
    index.add(vectors)
    return index


def embedd_word(text: str,
                vectorizer: WordVectorizerConfig,
                model: nn.Module) -> np.ndarray:
    model.eval()
    text_vectorized = vectorize_word(clean_phrase(text), num_features=vectorizer.num_features,
                                     max_length=vectorizer.max_length)
    text_t = torch.tensor([text_vectorized])
    with torch.no_grad():
        text_embedded = model(text_t).detach().numpy()
    return text_embedded


def prepare_model(
        model_config: ModelConfig,
        model_path: str
) -> nn.Module:
    model = SimpleTextEncoder(num_embeddings=model_config.num_embeddings, embedding_dim=256, fc_dim=128, output_dim=128)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def load_items_pool(candidates_path: str) -> Dict[int, Item]:
    idx = 0
    items = {}
    with open(candidates_path) as f:
        for line in f:
            item = json.loads(line)
            items[idx] = Item(business_id=item["product_id"], text=item["product_name"])
            idx += 1
    return items


def load_queries(queries_path: str) -> List[Query]:
    queries = []
    with open(queries_path) as f:
        for line in f:
            query_parsed = json.loads(line)
            query = Query(text=query_parsed["query_search_phrase"],
                          relevant_business_item_ids=query_parsed["relevant_product_ids"])
            queries.append(query)
    return queries


def execute_query(query_embedding: np.ndarray, similarity_metric: SimilarityMetric, index, k: int) -> List[int]:
    if similarity_metric == similarity_metric.COSINE:
        faiss.normalize_L2(query_embedding)
    distances, ids = index.search(query_embedding, k)
    return ids[0, :].tolist()


class RecallValidator:
    def __init__(self, model_config: ModelConfig,
                 candidates_path: str,
                 queries_path: str,
                 vectorizer_config: WordVectorizerConfig,
                 similarity_metric: SimilarityMetric,
                 k: int):
        self.model_config = model_config
        self.candidates_path = candidates_path
        self.queries_path = queries_path
        self.vectorizer_config = vectorizer_config
        self.similarity_metric = similarity_metric
        self.k = k

    def validate(self, model: nn.Module) -> float:
        print("Starting recall validation")
        items = load_items_pool(self.candidates_path)
        item_embeddings = {}
        aux_ids_by_product_id = {}
        for aux_id, item in items.items():
            if aux_id % 50000 == 0:
                print(f"{aux_id} items embedded")
            item_embedding = embedd_word(item.text, self.vectorizer_config, model)
            item_embeddings[aux_id] = EmbeddedItem(
                business_id=item.business_id,
                text=item.text,
                embedding=item_embedding
            )
            aux_ids_by_product_id[item.business_id] = aux_id
        all_item_embeddings = np.array([e.embedding[0] for e in item_embeddings.values()])
        index = create_faiss_index(
            dim=self.model_config.embedding_dim,
            vectors=all_item_embeddings,
            similarity_metric=self.similarity_metric
        )

        query_embeddings = []
        queries = load_queries(self.queries_path)
        for query in queries:
            query_embedded = embedd_word(query.text, self.vectorizer_config, model)
            query_embeddings.append(QueryEmbedded(
                text=query.text,
                relevant_aux_ids=[aux_ids_by_product_id[id] for id in query.relevant_business_item_ids],
                embedding=query_embedded
            ))

        recalls = []
        for idx, q in enumerate(query_embeddings):
            if idx % 1000 == 0:
                print(f"{idx} queries executed")
            query_result = execute_query(q.embedding, self.similarity_metric, index, self.k)
            intersection = set(query_result).intersection(set(q.relevant_aux_ids))
            num_found = len(intersection)
            recall = num_found / self.k
            recalls.append(recall)
        recall = np.mean(np.array(recalls))[0]

        print(f"Validation recall: {recall}")
        return recall


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    model_path = 'bucket/gpu_setup/run_1/models/model_weights.pth'

    model_config = ModelConfig(num_embeddings=50000, embedding_dim=128)

    validator = RecallValidator(
        model_config=model_config,
        candidates_path='datasets/valid_candidates.json',
        queries_path='datasets/valid_queries.json',
        vectorizer_config=WordVectorizerConfig(max_length=10, num_features=50000),
        similarity_metric=SimilarityMetric.COSINE,
        k=1024)

    model = prepare_model(model_config, model_path)

    validator.validate(model)

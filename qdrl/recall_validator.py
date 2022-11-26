import json
import os
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import default_collate

from qdrl.configs import SimilarityMetric, Features
from qdrl.models import TwoTower, SimpleTextEncoder, MultiModalTwoTower
from qdrl.preprocess import clean_phrase, TextVectorizer
from qdrl.setup import setup_model, setup_vectorizer, parse_features
from qdrl.typo_generator import TypoGenerator


def prepare_model(
        model: nn.Module,
        model_path: str,
        from_checkpoint: bool = False
) -> nn.Module:
    model_state = torch.load(model_path, map_location=torch.device('cpu'))[
        "model_state_dict"] if from_checkpoint else torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    model.eval()
    return model


def load_candidates(candidates_path: str) -> Dict[int, Dict]:
    idx = 0
    candidates = {}
    with open(candidates_path) as f:
        for line in f:
            candidates[idx] = json.loads(line)
            idx += 1
    return candidates


def vectorize_text(vectorizer: TextVectorizer, text: str, typo_generator: Optional[TypoGenerator]) -> List[int]:
    text = clean_phrase(text)
    if typo_generator:
        text = typo_generator.create_typos(text)
    return vectorizer.vectorize(text)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def embed_candidates(
        candidates: List[Dict],
        model: nn.Module,
        batch_size: int,
        device: torch.device
):
    model.eval()
    batch_idx = 0
    embeddings = []
    batches = len(candidates) / batch_size
    checkpoints = [batches // 10 * i for i in range(0, 10)]
    for b in batch(candidates, batch_size):
        if batch_idx in checkpoints:
            print(f"Embedded {(batch_idx / batches) * 100} % candidate batches")
        collated = default_collate(b)
        if isinstance(model, SimpleTextEncoder):
            embedded = model.forward(collated[model.document_text_feature].to(device))
        elif isinstance(model, TwoTower):
            embedded = model.forward_document(collated[model.document_text_feature].to(device))
        elif isinstance(model, MultiModalTwoTower):
            embedded = model.forward_document(text=collated[model.document_text_feature].to(device),
                                             category=collated[model.document_categorical_feature].to(device))
        else:
            raise ValueError("Model type not supported")
        batch_idx += 1
        embeddings.append(embedded.detach().cpu().numpy())
    return np.vstack(embeddings)


def embed_queries(
        queries: List[Dict],
        model: nn.Module,
        batch_size: int,
        device: torch.device
):
    model.eval()
    batch_idx = 0
    embeddings = []
    batches = len(queries) / batch_size
    checkpoints = [batches // 10 * i for i in range(0, 10)]
    for b in batch(queries, batch_size):
        if batch_idx in checkpoints:
            print(f"Embedded {(batch_idx / batches) * 100} % query batches")
        collated = default_collate(b)
        if isinstance(model, SimpleTextEncoder):
            embedded = model.forward(collated[model.query_text_feature].to(device))
        elif isinstance(model, TwoTower):
            embedded = model.forward_query(collated[model.query_text_feature].to(device))
        elif isinstance(model, MultiModalTwoTower):
            embedded = model.forward_query(text=collated[model.query_text_feature].to(device))
        else:
            raise ValueError("Model type not supported")
        batch_idx += 1
        embeddings.append(embedded.detach().cpu().numpy())
    return np.vstack(embeddings)


def create_index(
        dim: int,
        embeddings: np.ndarray,
        similarity_metric: SimilarityMetric):
    print("Trying to create index")
    index = faiss.IndexFlat(dim, faiss.METRIC_INNER_document)
    if similarity_metric == SimilarityMetric.COSINE:
        faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def load_queries(queries_path: str) -> List[Dict]:
    queries = []
    with open(queries_path) as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def execute_query(query_embeddings: np.ndarray, similarity_metric: SimilarityMetric, index, k: int) -> [np.ndarray,
                                                                                                        np.ndarray]:
    if similarity_metric == similarity_metric.COSINE:
        faiss.normalize_L2(query_embeddings)
    distances, ids = index.search(query_embeddings, k)
    return distances, ids


def search(embeddings: np.ndarray, batch_size: int, similarity_metric: SimilarityMetric, index, k: int) -> np.ndarray:
    idx = 0
    cnt = embeddings.shape[0]
    query_results = []
    batches = cnt / batch_size
    batch_idx = 0
    checkpoints = [batches // 10 * i for i in range(0, 10)]

    while idx < cnt:
        if batch_idx in checkpoints:
            print(f"finished searching for {(batch_idx / batches) * 100} % batches")
        batch = embeddings[idx:idx + batch_size, :]
        _, ids = execute_query(batch, similarity_metric=similarity_metric, index=index, k=k)
        query_results.append(ids)
        idx += batch_size
        batch_idx += 1

    return np.vstack(query_results)


def parse_rows(
        rows: List[Dict],
        wanted_features: List[str],
        features: Features,
        vectorizer: TextVectorizer,
        typo_generator: Optional[TypoGenerator] = None) -> List[Dict]:
    parsed = []
    text_features = [f for f in wanted_features if f in features.text_features]
    categorical_features = [cf for cf in features.categorical_features if cf.name in wanted_features]
    for row in rows:
        c = {}
        for text_feature in text_features:
            c[text_feature] = np.array(vectorize_text(vectorizer, (row[text_feature]), typo_generator), dtype="int")
        for categorical_feature in categorical_features:
            c[categorical_feature.name] = np.array(categorical_feature.mapper.map(row[categorical_feature.name]),
                                                   dtype="int")
        parsed.append(c)
    return parsed


def candidates_index(candidates_path: str,
                     features: Features,
                     vectorizer: TextVectorizer,
                     embedding_dim: int,
                     embedding_batch_size: int,
                     similarity_metric: SimilarityMetric,
                     model: nn.Module,
                     device: torch.device):
    candidates = load_candidates(candidates_path)
    candidates_by_document_id = {v["document_id"]: k for k, v in candidates.items()}
    print("Loaded candidates")
    candidates_parsed = parse_rows(candidates.values(), features.document_features, features, vectorizer)
    print("Vectorized candidates")
    candidate_embeddings = embed_candidates(candidates_parsed, model, embedding_batch_size, device)
    print("Embedded candidates")
    index = create_index(embedding_dim, candidate_embeddings, similarity_metric)
    print("Created candidates index")
    return index, candidates_by_document_id, candidates, candidate_embeddings


def write_embeddings(logdir_path: str, candidates: Dict[int, Dict], candidate_embeddings: np.ndarray):
    indices = np.random.randint(low=0, high=len(candidates), size=10000)
    tensorboard_writer = SummaryWriter(log_dir=logdir_path)
    document_names = [v["document_name"] for k, v in candidates.items()]
    meta = np.array(document_names)[indices]
    tensorboard_writer.add_embedding(
        mat=candidate_embeddings[indices, :], metadata=meta
    )
    tensorboard_writer.close()


def filter_invalid_queries(query_results: np.ndarray, queries: List[Dict], candidates_by_document_id: Dict[str, Dict]) -> \
        List[Tuple[np.ndarray, List[int]]]:
    missing_candidates = set()
    invalid_queries = 0
    query_results_with_relevant_items = []

    for idx, query_result in enumerate(query_results.tolist()):
        relevant_candidate_ids = queries[idx]["relevant_document_ids"]
        relevant_aux_ids = []
        for relevant_candidate_id in relevant_candidate_ids:
            if relevant_candidate_id not in candidates_by_document_id:
                missing_candidates.add(relevant_candidate_id)
            else:
                relevant_aux_ids.append(candidates_by_document_id[relevant_candidate_id])
        if relevant_aux_ids:
            query_results_with_relevant_items.append(
                (query_result, relevant_aux_ids)
            )
        else:
            invalid_queries += 1

    print(f"Number of candidates not found in recall validation : {len(missing_candidates)}")
    print(f"Number of skipped queries in recall validation: {invalid_queries}")
    return query_results_with_relevant_items


def calculate_recall(query_results_with_relevant_items: List[Tuple[np.ndarray, List[int]]], k: int) -> float:
    recalls = []
    for query_result, relevant_aux_ids in query_results_with_relevant_items:
        qr_at_k = query_result[:k]
        intersection = set(qr_at_k).intersection(set(relevant_aux_ids))
        num_found = len(intersection)
        recall = num_found / len(relevant_aux_ids)
        recalls.append(recall)
    recall = np.mean(np.array(recalls))
    return recall


def calculate_mrr(query_results_with_relevant_items: List[Tuple[np.ndarray, List[int]]], k: int) -> float:
    reciprocal_ranks = []
    for query_result, relevant_aux_ids in query_results_with_relevant_items:
        qr_at_k = query_result[:k]
        found = False
        relevant_items_aux_ids = set(relevant_aux_ids)
        for item_rank, item_aux_id in enumerate(qr_at_k, 1):
            if item_aux_id in relevant_items_aux_ids:
                reciprocal_ranks.append(1 / item_rank)
                found = True
                break
        if not found:
            reciprocal_ranks.append(0.0)
    mrr = np.mean(np.array(reciprocal_ranks))
    return mrr


def interactive_search(candidates_path: str,
                       features: Features,
                       vectorizer: TextVectorizer,
                       embedding_dim: int,
                       embedding_batch_size: int,
                       query_batch_size: int,
                       similarity_metric: SimilarityMetric,
                       k: int,
                       model: nn.Module,
                       device: torch.device):
    index, candidates_by_document_id, candidates, _ = candidates_index(
        candidates_path=candidates_path,
        features=features,
        vectorizer=vectorizer,
        embedding_dim=embedding_dim,
        embedding_batch_size=embedding_batch_size,
        similarity_metric=similarity_metric,
        model=model,
        device=device
    )

    while True:
        query = input("Type your query: \n")
        query_dict = {"query_search_phrase": query}
        query_parsed = parse_rows([query_dict], features.query_features, features, vectorizer)
        query_embeddings = embed_queries(query_parsed, model, batch_size=embedding_batch_size, device=device)
        query_results = search(query_embeddings, query_batch_size, similarity_metric, index, k)
        for query_result in query_results.tolist():
            for result_idx, result in enumerate(query_result):
                print(f"{result_idx}: {candidates[result]['document_name']}")


def recall_validation(
        candidates_path: str,
        queries_path: str,
        features: Features,
        vectorizer: TextVectorizer,
        embedding_dim: int,
        embedding_batch_size: int,
        query_batch_size: int,
        similarity_metric: SimilarityMetric,
        ks: List[int],
        model: nn.Module,
        device: torch.device,
        query_typo_probabilities: List[float],
        visualize_path: Optional[str] = None
) -> Dict[str, float]:
    index, candidates_by_document_id, candidates, candidate_embeddings = candidates_index(
        candidates_path=candidates_path,
        features=features,
        vectorizer=vectorizer,
        embedding_dim=embedding_dim,
        embedding_batch_size=embedding_batch_size,
        similarity_metric=similarity_metric,
        model=model,
        device=device
    )

    queries = load_queries(queries_path)
    print("Loaded queries")
    metrics = {}
    for query_typo_probability in query_typo_probabilities:
        tg = TypoGenerator(query_typo_probability)
        metric_suffix = "" if query_typo_probability == 0.0 else f"/typos_prob={query_typo_probability}"
        queries_parsed = parse_rows(queries, features.query_features, features, vectorizer, tg)
        print("Vectorized queries")
        query_embeddings = embed_queries(queries_parsed, model, batch_size=embedding_batch_size, device=device)
        print("Embedded queries")
        query_results = search(query_embeddings, query_batch_size, similarity_metric, index, k=max(ks))
        print(f"Executed queries, will calculate validation metrics...")
        query_results_with_relevant_items = filter_invalid_queries(query_results, queries, candidates_by_document_id)
        for k in ks:
            recall = calculate_recall(query_results_with_relevant_items, k=k)
            metrics[f"Recall@{k}{metric_suffix}"] = recall
            print(f"Average recall@{k}{metric_suffix}: {recall}")

            mrr = calculate_mrr(query_results_with_relevant_items, k=k)
            metrics[f"MRR@{k}{metric_suffix}"] = mrr
            print(f"MRR@{k}{metric_suffix}: {mrr}")
    if visualize_path:
        write_embeddings(visualize_path, candidates, candidate_embeddings)
        print("Saved embedding visualization")

    return metrics


class RecallValidator:
    def __init__(self,
                 candidates_path: str,
                 queries_path: str,
                 features: Features,
                 vectorizer: TextVectorizer,
                 embedding_dim: int,
                 embedding_batch_size: int,
                 query_batch_size: int,
                 similarity_metric: SimilarityMetric,
                 k: List[int],
                 query_typo_probabilities: List[float],
                 device: torch.device,
                 ):
        self.candidates_path = candidates_path
        self.queries_path = queries_path
        self.embedding_dim = embedding_dim
        self.embedding_batch_size = embedding_batch_size
        self.query_batch_size = query_batch_size
        self.similarity_metric = similarity_metric
        self.k = k
        self.device = device
        self.vectorizer = vectorizer
        self.features = features
        self.query_typo_probabilities = query_typo_probabilities

    def validate(self, model: nn.Module) -> Dict[str, float]:
        return recall_validation(
            candidates_path=self.candidates_path,
            queries_path=self.queries_path,
            embedding_dim=self.embedding_dim,
            similarity_metric=self.similarity_metric,
            model=model,
            embedding_batch_size=self.embedding_batch_size,
            ks=self.k,
            query_batch_size=self.query_batch_size,
            device=self.device,
            vectorizer=self.vectorizer,
            features=self.features,
            query_typo_probabilities=self.query_typo_probabilities
        )


def setup_recall_validator(config: DictConfig, vectorizer: TextVectorizer, device: torch.device, features: Features) -> \
        Optional[RecallValidator]:
    if not config.recall_validation.enabled:
        print("Skipping recall validation")
        return None
    else:
        return RecallValidator(
            candidates_path=os.path.join(config.dataset_dir, "recall_validation_items_dataset", "items.json"),
            queries_path=os.path.join(config.dataset_dir, "recall_validation_queries_dataset", "queries.json"),
            vectorizer=vectorizer,
            embedding_dim=config.model.output_dim,
            similarity_metric=SimilarityMetric.COSINE,
            embedding_batch_size=4096,
            k=config.recall_validation.k,
            query_batch_size=128,
            device=device,
            features=features,
            query_typo_probabilities=config.recall_validation.query_typo_probabilities
        )


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    config_file = "models/loss_function/config.yaml"
    model_path = 'models/loss_function/models/model_weights.pth'

    candidates_path = 'datasets/local/longest/recall_validation_items_dataset/items.json'
    queries_path = 'datasets/local/longest/recall_validation_queries_dataset/queries.json'

    conf = OmegaConf.load(config_file)

    model = setup_model(conf)
    vectorizer = setup_vectorizer(conf)

    model = prepare_model(model, model_path, from_checkpoint=False)
    features = parse_features(conf)

    recall_validation(candidates_path,
                      queries_path,
                      features=features,
                      vectorizer=vectorizer,
                      embedding_dim=128,
                      similarity_metric=SimilarityMetric.COSINE,
                      model=model,
                      embedding_batch_size=4096,
                      ks=[10, 50, 100, 500, 1000],
                      query_batch_size=128,
                      visualize_path="tensorboard/embeddings",
                      device=torch.device("cpu"),
                      query_typo_probabilities=[0.0]
                      )

    # interactive_search(
    #     candidates_path,
    #     embedding_dim=conf.model.output_dim,
    #     similarity_metric=SimilarityMetric.COSINE,
    #     model=model,
    #     embedding_batch_size=4096,
    #     k=30,
    #     query_batch_size=128,
    #     device=torch.device("cpu"),
    #     vectorizer=vectorizer,
    #     features=features
    # )

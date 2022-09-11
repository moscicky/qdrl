import json
import os
from typing import Dict, List, Optional

import faiss
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from qdrl.configs import SimilarityMetric, ModelConfig
from qdrl.models import SimpleTextEncoder
from qdrl.preprocess import clean_phrase, TextVectorizer, WordUnigramVectorizer, DictionaryLoaderTextVectorizer


def prepare_model(
        model_config: ModelConfig,
        model_path: str,
        from_checkpoint: bool = False
) -> nn.Module:
    model = SimpleTextEncoder(num_embeddings=model_config.num_embeddings, embedding_dim=256, fc_dim=128, output_dim=128)
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


def vectorize_text(vectorizer: TextVectorizer, text: str) -> List[int]:
    return vectorizer.vectorize(clean_phrase(text))


def vectorize(texts: List[str],
              vectorizer: TextVectorizer) -> np.ndarray:
    vectorized = []
    for text in texts:
        vectorized.append(
            np.array(vectorize_text(vectorizer, text), dtype=np.int32))
    return np.array(vectorized)


def embed(texts: np.ndarray, model: nn.Module, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    idx = 0
    cnt = texts.shape[0]
    embeddings = []
    batches = cnt / batch_size
    batch_idx = 0
    checkpoints = [batches // 10 * i for i in range(0, 10)]

    while idx < cnt:
        if batch_idx in checkpoints:
            print(f"processed {(batch_idx / batches) * 100} % batches")
        batch = texts[idx:idx + batch_size, :]
        batch_tensor = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            text_embedded = model(batch_tensor).detach().cpu().numpy()
        embeddings.append(text_embedded)
        idx += batch_size
        batch_idx += 1

    return np.vstack(embeddings)


def create_index(
        dim: int,
        embeddings: np.ndarray,
        similarity_metric: SimilarityMetric):
    print("Trying to create index")
    index = faiss.IndexFlat(dim, faiss.METRIC_INNER_PRODUCT)
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
            print(f"processed {(batch_idx / batches) * 100} % batches")
        batch = embeddings[idx:idx + batch_size, :]
        _, ids = execute_query(batch, similarity_metric=similarity_metric, index=index, k=k)
        query_results.append(ids)
        idx += batch_size
        batch_idx += 1

    return np.vstack(query_results)


def candidates_index(candidates_path: str, vectorizer: TextVectorizer, embedding_dim: int,
                     embedding_batch_size: int, similarity_metric: SimilarityMetric, model: nn.Module,
                     device: torch.device):
    candidates = load_candidates(candidates_path)
    candidates_by_product_id = {v["product_id"]: k for k, v in candidates.items()}
    print("Loaded candidates")
    candidates_vectorized = vectorize([c["product_name"] for c in candidates.values()], vectorizer)
    print("Vectorized candidates")
    candidate_embeddings = embed(candidates_vectorized, model, batch_size=embedding_batch_size, device=device)
    print("Embedded candidates")
    index = create_index(embedding_dim, candidate_embeddings, similarity_metric)
    print("Created candidates index")
    return index, candidates_by_product_id, candidates, candidate_embeddings


def write_embeddings(logdir_path: str, candidates: Dict[int, Dict], candidate_embeddings: np.ndarray):
    indices = np.random.randint(low=0, high=len(candidates), size=10000)
    tensorboard_writer = SummaryWriter(log_dir=logdir_path)
    product_names = [v["product_name"] for k, v in candidates.items()]
    meta = np.array(product_names)[indices]
    tensorboard_writer.add_embedding(
        mat=candidate_embeddings[indices, :], metadata=meta
    )
    tensorboard_writer.close()


def calculate_recall(query_results: np.ndarray, queries: List[Dict],
                     candidates_by_product_id: Dict[str, Dict]) -> float:
    recalls = []
    missing_counter = 0

    for idx, query_result in enumerate(query_results.tolist()):
        relevant_candidate_ids = queries[idx]["relevant_product_ids"]
        relevant_aux_ids = []
        for relevant_candidate_id in relevant_candidate_ids:
            if relevant_candidate_id not in candidates_by_product_id:
                missing_counter += 1
                continue
            else:
                relevant_aux_ids.append(candidates_by_product_id[relevant_candidate_id])
        if relevant_aux_ids:
            intersection = set(query_result).intersection(set(relevant_aux_ids))
            num_found = len(intersection)
            # if num_found == 0:
            #     print(f"Not found any products: {queries[idx]['query_search_phrase']}")
            recall = num_found / len(relevant_aux_ids)
            recalls.append(recall)
    print(f"Number of candidates not found during recall validation: {missing_counter}")
    recall = np.mean(np.array(recalls))
    return recall


def interactive_search(candidates_path: str,
                       vectorizer: TextVectorizer,
                       num_embeddings: int,
                       embedding_dim: int,
                       embedding_batch_size: int,
                       query_batch_size: int,
                       similarity_metric: SimilarityMetric,
                       k: int,
                       model: nn.Module,
                       device: torch.device):
    index, candidates_by_product_id, candidates, _ = candidates_index(
        candidates_path=candidates_path,
        vectorizer=vectorizer,
        embedding_dim=embedding_dim,
        embedding_batch_size=embedding_batch_size,
        similarity_metric=similarity_metric,
        model=model,
        device=device
    )

    while True:
        query = input("Type your query: \n")
        query_vectorized = vectorize([query], vectorizer)
        query_embeddings = embed(query_vectorized, model, batch_size=embedding_batch_size, device=device)
        query_results = search(query_embeddings, query_batch_size, similarity_metric, index, k)
        for query_result in query_results.tolist():
            for result_idx, result in enumerate(query_result):
                print(f"{result_idx}: {candidates[result]['product_name']}")


def recall_validation(
        candidates_path: str,
        queries_path: str,
        vectorizer: TextVectorizer,
        embedding_dim: int,
        embedding_batch_size: int,
        query_batch_size: int,
        similarity_metric: SimilarityMetric,
        k: int,
        model: nn.Module,
        device: torch.device,
        visualize_path: Optional[str] = None
):
    index, candidates_by_product_id, candidates, candidate_embeddings = candidates_index(
        candidates_path=candidates_path,
        vectorizer=vectorizer,
        embedding_dim=embedding_dim,
        embedding_batch_size=embedding_batch_size,
        similarity_metric=similarity_metric,
        model=model,
        device=device
    )

    queries = load_queries(queries_path)
    print("Loaded queries")
    queries_vectorized = vectorize([q["query_search_phrase"] for q in queries], vectorizer)
    print("Vectorized queries")
    query_embeddings = embed(queries_vectorized, model, batch_size=embedding_batch_size, device=device)
    print("Embedded queries")
    query_results = search(query_embeddings, query_batch_size, similarity_metric, index, k)
    print("Executed queries")
    recall = calculate_recall(query_results, queries, candidates_by_product_id)
    print(f"Average recall@{k}: {recall}")
    if visualize_path:
        write_embeddings(visualize_path, candidates, candidate_embeddings)
        print("Saved embedding visualization")

    return recall


class RecallValidator:
    def __init__(self,
                 candidates_path: str,
                 queries_path: str,
                 vectorizer: TextVectorizer,
                 embedding_dim: int,
                 embedding_batch_size: int,
                 query_batch_size: int,
                 similarity_metric: SimilarityMetric,
                 k: int,
                 device: torch.device
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

    def validate(self, model: nn.Module):
        return recall_validation(
            candidates_path=self.candidates_path,
            queries_path=self.queries_path,
            embedding_dim=self.embedding_dim,
            similarity_metric=self.similarity_metric,
            model=model,
            embedding_batch_size=self.embedding_batch_size,
            k=self.k,
            query_batch_size=self.query_batch_size,
            device=self.device,
            vectorizer=self.vectorizer
        )


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    model_path = 'models/vectorizer_150k_0k_0k_150k/model.pth'
    candidates_path = 'datasets/local/recall_validation_items_dataset/items.json'
    queries_path = 'datasets/local/recall_validation_queries_dataset/queries.json'

    model_config = ModelConfig(num_embeddings=300000, embedding_dim=128)

    vectorizer = DictionaryLoaderTextVectorizer(
        dictionary_path="datasets/local/token_dictionary_150k_0k_0k",
        word_unigrams_limit=8,
        word_bigrams_limit=0,
        char_trigrams_limit=0,
        num_oov_tokens=150000)

    model = prepare_model(model_config, model_path, from_checkpoint=False)

    # recall_validation(candidates_path,
    #                   queries_path,
    #                   vectorizer=vectorizer,
    #                   embedding_dim=128,
    #                   similarity_metric=SimilarityMetric.COSINE,
    #                   model=model,
    #                   embedding_batch_size=4096,
    #                   k=1024,
    #                   query_batch_size=128,
    #                   visualize_path="tensorboard/embeddings",
    #                   device=torch.device("cpu")
    #                   )

    interactive_search(
        candidates_path,
        num_embeddings=model_config.num_embeddings,
        embedding_dim=128,
        similarity_metric=SimilarityMetric.COSINE,
        model=model,
        embedding_batch_size=4096,
        k=30,
        query_batch_size=128,
        device=torch.device("cpu"),
        vectorizer=vectorizer
    )

import json
from typing import Dict, List

import numpy as np
import faiss
import torch

from qdrl.main_train import NeuralNet
from qdrl.preprocess import vectorize_word

if __name__ == '__main__':
    dim = 64
    index = faiss.IndexFlatL2(dim)
    candidates: Dict[int, Dict] = {}
    candidates_by_product_id: Dict[str, int] = {}
    counter = 0
    num_embeddings = 10000

    queries: List[Dict] = []

    model = NeuralNet(num_embeddings=num_embeddings, embedding_dim=dim)
    model.load_state_dict(torch.load('models/model_weights.pth'))
    model.eval()

    with open('datasets/valid_candidates.json') as file:
        for line in file:
            if counter % 50000 == 0:
                print(f"{counter} products embedded")
            candidate = json.loads(line)
            product_name = vectorize_word(candidate["product_name"], num_embeddings, 10)
            product_name_t = torch.tensor([product_name])
            product_embedded = model(product_name_t).detach().numpy()[:1, :]
            candidates[counter] = {
                "id": candidate["product_id"],
                "name": candidate["product_name"],
                "embedding": product_embedded
            }
            candidates_by_product_id[candidate["product_id"]] = counter
            index.add(product_embedded)
            counter += 1

    with open('datasets/valid_queries.json') as file:
        for line in file:
            query = json.loads(line)
            query_vectorized = vectorize_word(query["query_search_phrase"], num_embeddings, 10)
            query_t = torch.tensor([query_vectorized])
            query_embedded = model(product_name_t).detach().numpy()[:1, :]
            q = {
                "search_phrase": query["query_search_phrase"],
                "relevant_product_ids": [candidates_by_product_id[i] for i in query["relevant_product_ids"]],
                "embedding": query_embedded
            }
            queries.append(q)

    k = 1024

    recalls = []
    for idx, query in enumerate(queries):
        if idx % 1000 == 0:
            print(f"{idx} queries done")
        D, I = index.search(query["embedding"], k)
        neighs = I[0, :].tolist()
        relevant_product_ids = query["relevant_product_ids"]
        intersection = set(relevant_product_ids).intersection(set(neighs))
        found_cnt = len(intersection)
        recall = found_cnt / k
        recalls.append(recall)

    print(np.mean(np.array(recalls)))
    print("Done!")

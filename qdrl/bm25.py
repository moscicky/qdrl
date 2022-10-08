from typing import Dict, Any, List

from elasticsearch import Elasticsearch, helpers

from qdrl.preprocess import clean_phrase
from qdrl.recall_validator import load_candidates, load_queries, calculate_recall, calculate_mrr


def clean_candidates(candidates: List[Dict[str, Any]]):
    return [{**c, "product_name": clean_phrase(c["product_name"])} for c in candidates]


def clean_queries(queries: List[Dict[str, Any]]):
    return [{**q, "query_search_phrase": clean_phrase(q["query_search_phrase"])} for q in queries]


es_index = "candidates"


def insert_candidates(candidates: List[Dict[str, Any]]):
    actions = []
    for candidate in candidates:
        action = {
            "_index": es_index,
            "doc_type": "candidate",
            "_id": candidate["product_id"],
            "_source": {
                "id": candidate["product_id"],
                "product_name": candidate["product_name"],
            }
        }
        actions.append(action)

    helpers.bulk(es, actions, refresh=True)


def execute_queries(es: Elasticsearch, queries: List[Dict[str, Any]], k: int):
    results = []
    batch_size = 100
    l = len(queries)
    for ndx in range(0, l, batch_size):
        if ndx % 1000 == 0:
            print(f"Executed {ndx} queries")
        request = []
        for query in queries[ndx:min(ndx + batch_size, l)]:
            req_head = {'index': es_index}
            req_body = {
                "query": {
                    "match": {"product_name": query["query_search_phrase"]}
                },
                "size": k,
                "_source": ["_id"],
            }
            request.extend([req_head, req_body])
        res = es.msearch(body=request)
        tmp = [[h['_id'] for h in r['hits']['hits']] for r in res.body['responses']]
        results.extend(tmp)
    ret = []
    for q, r in zip(queries, results):
        ret.append((r, q["relevant_product_ids"]))
    return ret


def search(
        candidates_path: str,
        queries_path: str,
        es: Elasticsearch,
        cleanup: bool,
        load: bool,
        ks: List[int]
):
    if cleanup:
        es.options(ignore_status=[400, 404]).indices.delete(index=es_index)
    candidates = load_candidates(candidates_path).values()
    candidates = clean_candidates(candidates)

    if load:
        insert_candidates(candidates)
    queries = load_queries(queries_path)
    queries = clean_queries(queries)

    query_results_with_relevant_items = []

    execute_queries(es, queries, max(ks))

    # checkpoints = [len(queries) // 10 * i for i in range(0, 10)]
    # for idx, query in enumerate(queries):
    #     if idx in checkpoints:
    #         print(f"finished searching for {(idx / len(queries)) * 100} % queries")
    #     resp = es.search(index=es_index, query={"match": {"product_name": query["query_search_phrase"]}},
    #                      source=["_id"], size=max(ks))
    #     found_ids = [h["_id"] for h in resp.body['hits']['hits']]
    #     query_results_with_relevant_items.append((found_ids, query["relevant_product_ids"]))
    query_results_with_relevant_items = execute_queries(es, queries, k=max(ks))
    print("executed queries")

    # TODO: filter out invalid queries

    for k in ks:
        recall = calculate_recall(query_results_with_relevant_items, k=k)
        print(f"Average recall@{k}: {recall}")

        mrr = calculate_mrr(query_results_with_relevant_items, k=k)
        print(f"MRR@{k}: {mrr}")


if __name__ == '__main__':
    es = Elasticsearch("https://localhost:9200",
                        timeout=10000)

    candidates_path = 'datasets/local/longest/recall_validation_items_dataset/items.json'
    queries_path = 'datasets/local/longest/recall_validation_queries_dataset/queries.json'

    search(
        candidates_path=candidates_path,
        queries_path=queries_path,
        es=es,
        cleanup=False,
        load=False,
        ks=[10, 60, 100, 1024]
    )

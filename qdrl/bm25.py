import argparse
from typing import Dict, Any, List, Tuple

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


def execute_queries(es: Elasticsearch, queries: List[Dict[str, Any]], k: int) -> List[Tuple[List[str], List[str]]]:
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
        res = es.msearch(searches=request)
        tmp = [[h['_id'] for h in r['hits']['hits']] for r in res.body['responses']]
        results.extend(tmp)
    ret = []
    for q, r in zip(queries, results):
        ret.append((r, q["relevant_product_ids"]))
    return ret


def filter_invalid_queries(queries: List[Dict], candidates: List[Dict]) -> List[Dict]:
    missing_candidates = set()
    invalid_queries = 0
    candidate_ids = set([c["product_id"] for c in candidates])

    filtered_queries = []

    for idx, query in enumerate(queries):
        relevant_candidate_ids = query["relevant_product_ids"]
        good_candidates = []
        for relevant_candidate_id in relevant_candidate_ids:
            if relevant_candidate_id not in candidate_ids:
                missing_candidates.add(relevant_candidate_id)
            else:
                good_candidates.append(relevant_candidate_id)
        if good_candidates:
            fixed_query = {**query, **{"relevant_product_ids": good_candidates}}
            filtered_queries.append(fixed_query)
        else:
            invalid_queries += 1

    print(f"Number of candidates not found in recall validation : {len(missing_candidates)}")
    print(f"Number of skipped queries in recall validation: {invalid_queries}")
    return filtered_queries


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
    queries = filter_invalid_queries(queries, candidates)
    query_results_with_relevant_items = execute_queries(es, queries, k=max(ks))
    print("executed queries")
    for k in ks:
        recall = calculate_recall(query_results_with_relevant_items, k=k)
        print(f"Average recall@{k}: {recall}")

        mrr = calculate_mrr(query_results_with_relevant_items, k=k)
        print(f"MRR@{k}: {mrr}")


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        '--cert-file-path',
        type=str,
        default=None,
        required=True
    )

    args_parser.add_argument(
        '--elastic-url',
        type=str,
        default="https://localhost:9200",
        required=False
    )

    args_parser.add_argument(
        '--elastic-user',
        type=str,
        default="elastic",
        required=False
    )

    args_parser.add_argument(
        '--elastic-password',
        type=str,
        default=None,
        required=True
    )

    options = args_parser.parse_args()

    es = Elasticsearch(options.elastic_url,
                       ca_certs=options.cert_file_path,
                       basic_auth=(options.elastic_user, options.elastic_password), request_timeout=10000)

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

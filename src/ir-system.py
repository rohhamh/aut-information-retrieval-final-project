import json
import sys
import time
import pickle
import argparse
import hashlib
import heapq
import numpy as np
from tqdm import tqdm

from class_definitions import (
    DatasetDocumentToken,
    Index,
    Posting,
    DatasetDocument,
    QueryResult,
)
from tokenization_utils import get_string_tokens


def load_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        dataset_dict: dict = json.load(f)
    dataset: dict[str, DatasetDocument] = {}
    for key in dataset_dict.keys():
        dataset[key] = DatasetDocument(key, **dataset_dict[key])
    return dataset


def index_document(index: Index, document: DatasetDocument):
    for t in document.tokens:
        posting = Posting(
            document_id=document.id,
            term_freq=t.token_freq,
            term_positions=t.position_indices,
        )
        index.add(t.token, posting)


def index_dataset(dataset: dict[str, DatasetDocument]) -> tuple[Index, dict[str, int]]:
    blake2 = hashlib.blake2b()
    blake2.update(repr(dataset).encode())
    index = Index(dataset_hash=blake2.digest())
    for document in dataset.values():
        index_document(index, document)
    removed_indices = remove_top_k_frequest_indices(index, 15)
    return index, removed_indices


def process_dataset(dataset: dict[str, DatasetDocument]):
    for v in tqdm(dataset.values()):
        tokens_counts, tokens_positions = get_string_tokens(v.content)
        tokens = tokens_counts.keys()
        for token in tokens:
            count, positions = tokens_counts[token], tokens_positions[token]
            v.tokens.append(DatasetDocumentToken(token, positions, count))
    vectorize_dataset(dataset)


def remove_top_k_frequest_indices(index: Index, k: int):
    k_most_frequent_terms = heapq.nlargest(k, index.items(), key=lambda x: x[1].tf)
    removed_indices = dict()
    for mft, index_item in k_most_frequent_terms:
        index.remove(mft)
        removed_indices[mft] = index_item.tf
    return removed_indices


def weigh_token(token_freq: int, df: int) -> float:
    if token_freq == 0 or df == 0:
        return 0
    tf = 1 + np.log10(token_freq)
    idf = np.log10(len(dataset) / df)
    return tf * idf


def vectorize_dataset(dataset: dict[str, DatasetDocument]):
    for document in dataset.values():
        document.vector = list(
            map(
                lambda dtoken: (
                    dtoken.token,
                    weigh_token(dtoken.token_freq, 1),  # lnc
                ),
                document.tokens,
            )
        )
        document.vector = list(filter(lambda v: v[1] > 0, document.vector))


def get_vectors_similarity_score(
    v1: list[tuple[str, float]], v2: list[tuple[str, float]]
) -> float:
    dv1, dv2 = dict(v1), dict(v2)
    common_terms = dv1.keys() & dv2.keys()
    if not common_terms:
        return 0
    dv1 = {x: dv1[x] for x in common_terms}
    dv2 = {x: dv2[x] for x in common_terms}
    a = np.array(list(dv1.values()))
    b = np.array(list(dv2.values()))
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def answer_query(query: str, k: int, index: Index, dataset: dict[str, DatasetDocument]):
    qtokens_counts, _ = get_string_tokens(query)
    qtokens = qtokens_counts.keys()
    print(qtokens)
    qvector = list(
        map(
            lambda qtoken: (
                qtoken,
                weigh_token(
                    qtokens_counts[qtoken],
                    index.get(qtoken).df if index.has(qtoken) else 0,
                ),  # ltc
            ),
            qtokens,
        )
    )
    scores = [
        (v.id, get_vectors_similarity_score(qvector, v.vector))
        for v in dataset.values()
    ]
    best_matches = [
        i for i in heapq.nlargest(k, scores, key=lambda s: s[1]) if i[1] > 0
    ]
    results = []
    for m in best_matches:
        id, score = m
        document = dataset[id]
        results.append(QueryResult(document.id, document.title, document.url))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ir-system.py", description="search json dataset for given query"
    )
    parser.add_argument("--dataset", help="the dataset to index", required=False)
    parser.add_argument("-q", "--query", help="the query to search for", required=False)
    parser.add_argument(
        "--save-index", help="saves created index at given path", required=False
    )
    parser.add_argument("--index", help="use saved index", required=False)

    args = parser.parse_args()

    if not (args.save_index and args.dataset) and not (args.query and args.dataset):
        sys.stdout.write("invalid options")
        exit(1)

    if args.dataset:
        dataset = load_dataset(args.dataset)
        start = time.time()
        process_dataset(dataset)
        print(f"processed dataset in {time.time() - start}s")

    if args.index:
        with open(args.index, "rb") as index_file:
            index = pickle.load(index_file)
    else:
        index, removed_indices = index_dataset(dataset)
        print(f"removed_indices {removed_indices}")

    if args.save_index:
        with open(args.save_index, "wb") as save_file:
            pickle.dump(index, save_file, 0)

    if args.query:
        answer = answer_query(query=args.query, k=10, index=index, dataset=dataset)
        print(answer)

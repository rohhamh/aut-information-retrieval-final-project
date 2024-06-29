import json
import sys
import time
import pickle
import argparse
import hashlib
import heapq
import numpy as np
from tqdm import tqdm
from pprint import pprint

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


def index_dataset(
    dataset: dict[str, DatasetDocument],
    champions_list_size: int | None,
    remove_top_k: int = 50,
) -> tuple[Index, dict[str, int]]:
    blake2 = hashlib.blake2b()
    blake2.update(repr(dataset).encode())
    index = Index(dataset_hash=blake2.digest())
    for document in dataset.values():
        index_document(index, document)
    removed_indices = remove_top_k_frequest_indices(index, remove_top_k)
    if champions_list_size:
        create_champions_list(index, champions_list_size)
    return index, removed_indices


def create_champions_list(index: Index, size: int):
    for v in index.values():
        v.champtions_list = heapq.nlargest(size, v.postings, key=lambda x: x.term_freq)


def get_dataset_document_vectors(
    dataset: dict[str, DatasetDocument], index: Index | None = None, tf_only: bool = False
):
    for v in tqdm(dataset.values()):
        tokens_counts, tokens_positions = get_string_tokens(v.content)
        tokens = tokens_counts.keys()
        for token in tokens:
            count, positions = tokens_counts[token], tokens_positions[token]
            v.tokens.append(DatasetDocumentToken(token, positions, count))
    if index:
        return vectorize_dataset(dataset, index, tf_only)
    return vectorize_dataset(dataset, None, tf_only)


def remove_top_k_frequest_indices(index: Index, k: int):
    k_most_frequent_terms = heapq.nlargest(k, index.items(), key=lambda x: x[1].tf)
    removed_indices = dict()
    for mft, index_item in k_most_frequent_terms:
        index.remove(mft)
        removed_indices[mft] = index_item.tf
    return removed_indices


def weigh_token(token_freq: int, df: int, tf_only: bool = False) -> float:
    if token_freq == 0 or df == 0:
        return 0

    tf = 1 + np.log10(token_freq)
    if tf_only:
        return tf
    idf = np.log10(len(dataset) / df)
    return tf * idf


def vectorize_dataset(dataset: dict[str, DatasetDocument], index: Index | None, tf_only: bool = False):
    document_vectors: dict[str, list[tuple[str, float]]] = dict()
    for document in dataset.values():
        def get_token_weight(dtoken):
            df = index.get(dtoken.token).df if (index and index.has(dtoken.token)) else 1
            return (
                    dtoken.token,
                    weigh_token(
                        dtoken.token_freq,
                        df,
                        tf_only,
                    ),  # lnc
                )
        document_vectors[document.id] = list(map(get_token_weight, document.tokens))
        document_vectors[document.id] = list(
            filter(lambda v: v[1] > 0, document_vectors[document.id])
        )
    return document_vectors


def get_vectors_similarity_score(
    v1: list[tuple[str, float]], v2: list[tuple[str, float]]
) -> float:
    dv1, dv2 = dict(v1), dict(v2)
    common_terms = dv1.keys() & dv2.keys()
    if not common_terms:
        return 0
    dv1_new = {x: dv1[x] for x in common_terms}
    dv2_new = {x: dv2[x] for x in common_terms}
    a = np.array(list(dv1_new.values()))
    b = np.array(list(dv2_new.values()))
    norm_a = np.linalg.norm(list(dv1.values()))
    norm_b = np.linalg.norm(list(dv2.values()))
    if not norm_a or not norm_b:
        return 0
    similarity = np.dot(a, b) / (norm_a * norm_b)
    return similarity


def answer_query(
    query: str,
    k: int,
    index: Index,
    dataset: dict[str, DatasetDocument],
    use_champions_list: bool = False,
) -> list[QueryResult]:
    qtokens_counts, _ = get_string_tokens(query)
    qtokens = qtokens_counts.keys()
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
    if use_champions_list:
        print('using champions_list')
        documents = list(
            set(
                [
                    champ for t in qtokens
                    for champ in index.get(t).champtions_list
                    if index.has(t)
                ]
            )
        )
        documents = [d.document_id for d in documents]
    else:
        documents = [d.id for d in dataset.values()]
    scores = [
        (id, get_vectors_similarity_score(qvector, index.document_vectors[id]))
        for id in documents
    ]
    best_matches = [
        i for i in heapq.nlargest(k, scores, key=lambda s: s[1]) if i[1] > 0
    ]
    results: list[QueryResult] = []
    for m in best_matches:
        id, score = m
        document = dataset[id]
        results.append(QueryResult(document.id, document.title, document.url, score))
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
    parser.add_argument(
        "--remove-top-indices",
        type=int,
        help="number of most frequent indices to remove",
        required=False,
    )
    parser.add_argument(
        "--n-results", type=int, help="number of top results to show", required=False
    )
    parser.add_argument(
        "--champions-list",
        action="store_true",
        help="use champions list",
        required=False,
    )
    parser.add_argument(
        "--champions-size",
        type=int,
        help="number of documents held as champions",
        required=False,
    )
    parser.add_argument(
        "--tf-only",
        action='store_true',
        help="use only tf for scoring",
        required=False,
    )

    args = parser.parse_args()

    if not (args.save_index and args.dataset) and not (args.query and args.dataset):
        sys.stdout.write("invalid options")
        exit(1)

    index: Index | None = None
    document_vectors: dict[str, list[tuple[str, float]]] | None = None

    if args.index:
        with open(args.index, "rb") as index_file:
            start = time.time()
            index = pickle.load(index_file)
            print(f"loading index took {time.time() - start}s")
            document_vectors = index.document_vectors  # type: ignore

    if args.dataset:
        start = time.time()
        dataset = load_dataset(args.dataset)
        if not document_vectors:
            if index:
                document_vectors = get_dataset_document_vectors(dataset, index, tf_only=args.tf_only)
            else:
                document_vectors = get_dataset_document_vectors(dataset, tf_only=args.tf_only)
            if index:
                index.document_vectors = document_vectors
        print(f"processed dataset in {time.time() - start}s")

    if not index:
        k = args.remove_top_indices or 50
        csize = args.champions_size or 20
        start = time.time()
        index, removed_indices = index_dataset(
            dataset, remove_top_k=k, champions_list_size=csize
        )
        print(f"indexing took {time.time() - start}s")
        if document_vectors:
            index.document_vectors = document_vectors
        print(f"removed_indices {removed_indices}")

    if args.save_index:
        with open(args.save_index, "wb") as save_file:
            pickle.dump(index, save_file, 0)

    if args.query:
        k = args.n_results or 10
        answer = answer_query(
            query=args.query,
            k=k,
            index=index,
            dataset=dataset,
            use_champions_list=args.champions_list,
        )
        pprint(answer)

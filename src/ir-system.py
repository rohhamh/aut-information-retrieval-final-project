import json
import sys
import time
import pickle
import argparse
import hashlib

from class_definitions import DatasetDocumentToken, Index, Posting, DatasetDocument
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


def index_dataset(dataset: dict[str, DatasetDocument]) -> Index:
    blake2 = hashlib.blake2b()
    blake2.update(repr(dataset).encode())
    index = Index(dataset_hash=blake2.digest())
    for document in dataset.values():
        index_document(index, document)
    return index


def process_dataset(dataset: dict[str, DatasetDocument]) -> Index:
    for v in dataset.values():
        tokens, tokens_counts, tokens_positions = get_string_tokens(v.content)
        for token in tokens:
            count, positions = tokens_counts[token], tokens_positions[token]
            v.tokens.append(DatasetDocumentToken(token, positions, count))
    index = index_dataset(dataset)
    return index


def answer_query(index: Index, query: str):
    qtokens, _, _ = get_string_tokens(query)


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

    if (
        not (args.save_index and args.dataset)
        and not (args.query and args.index)
        and not (args.query and args.dataset)
    ):
        sys.stdout.write("invalid options")
        exit(1)

    if args.index:
        with open(args.index, "rb") as index_file:
            index = pickle.load(index_file)

    if args.dataset:
        dataset = load_dataset(args.dataset)
        start = time.time()
        index = process_dataset(dataset)
        print(f"processed dataset in {time.time() - start}s")
        print(repr(dataset))
        exit(0)

    if args.save_index:
        with open(args.save_index, "wb") as save_file:
            pickle.dump(index, save_file, 0)

    if args.query:
        answer_query(index, args.query)

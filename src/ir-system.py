import json
import sys
import re
import time
import pickle
import argparse
import hashlib

from hazm import Normalizer, Stemmer, WordTokenizer

from class_definitions import DatasetDocumentToken, Index, Posting, DatasetDocument


def load_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        dataset_dict: dict = json.load(f)
    dataset: dict[str, DatasetDocument] = {}
    for key in dataset_dict.keys():
        dataset[key] = DatasetDocument(key, **dataset_dict[key])
    return dataset


def normalize(text: str):
    n = Normalizer()
    return n.normalize(text)


def tokenize(text: str):
    # tokenizer = WordTokenizer(
    #     join_verb_parts=False,
    #     join_abbreviations=False,
    #     replace_emails=True,
    #     replace_ids=True,
    #     replace_links=True,
    #     replace_numbers=True,
    # )
    # tokenizer.id_pattern = re.compile(r"(?<![\w._])(@[\w_-]+)")
    # tokens = tokenizer.tokenize(text)
    tokens_counts_indices: dict[str, list[int]] = {}
    positions: dict[str, list[int]] = {}
    pattern = re.compile(r'([؟!?]+|[\d.:]+|[:.،؛»\])}"«\[({/\\])')  # TODO \d
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f4cc\U0001f4cd"  # other emojis
        "]",
        flags=re.UNICODE,
    )
    emoji_repl = r"\g<0> "
    id_pattern = re.compile(r"(?<![\w._])(@[\w_-]+)")
    id_repl = r" ID "
    link_pattern = re.compile(
        r"((https?|ftp)://)?(?<!@)(([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})[-\w@:%_.+/~#?=&]*",
    )
    link_repl = r" LINK "
    email_pattern = re.compile(
        r"[a-zA-Z0-9._+-]+@([a-zA-Z0-9-]+\.)+[A-Za-z]{2,}",
    )
    email_repl = r" EMAIL "

    # '٫' is the decimal separator and '٬' is the thousands separator
    number_int_pattern = re.compile(
        r"\b(?<![\d۰-۹][.٫٬,])([\d۰-۹]+)(?![.٫٬,][\d۰-۹])\b",
    )
    number_int_repl = lambda m: " NUM" + str(len(m.group(1))) + " "
    number_float_pattern = re.compile(
        r"\b(?<!\.)([\d۰-۹,٬]+[.٫٬][\d۰-۹]+)\b(?!\.)",
    )
    number_float_repl = r" NUMF "
    text = email_pattern.sub(email_repl, text)
    text = link_pattern.sub(link_repl, text)
    text = id_pattern.sub(id_repl, text)
    text = number_int_pattern.sub(number_int_repl, text)
    text = number_float_pattern.sub(number_float_repl, text)

    text = pattern.sub(r" \1 ", text.replace("\n", " ").replace("\t", " "))

    tokens = [word for word in text.split(" ") if word]

    tokens = list(filter(lambda t: not re.match(r"(EMAIL)|(NUM\d*)|(LINK)", t), tokens))
    tokens = list(filter(lambda t: not re.match(r"\b\d+\b", t), tokens))

    for t in tokens:
        if t not in tokens_counts_indices:
            tokens_counts_indices[t] = [0, -1]
            positions[t] = []
        tokens_counts_indices[t][0] += 1
        re.search("rf\b{t}\b")
        idx = text.index(t, tokens_counts_indices[t][1] + 1)
        tokens_counts_indices[t][1] = idx
        positions[t].append(idx)
    return tokens, tokens_counts_indices, positions


def stem(terms: list[str]):
    stemmer = Stemmer()
    return list(map(stemmer.stem, terms))


def process_document(index: Index, document: DatasetDocument):
    for t in document.tokens:
        posting = Posting(
            document_id=document.id,
            term_freq=t.token_freq,
            term_positions=t.position_indices,
        )
        index.add(t.token, posting)


def index_dataset(dataset: dict[str, DatasetDocument]) -> Index:
    blake2 = hashlib.blake2b()
    blake2.digest
    blake2.update(repr(dataset).encode())
    index = Index(dataset_hash=blake2.digest())
    for document in dataset.values():
        process_document(index, document)
    return index


def merge_positional_indices(
    raw_tokens,
    stemmed_tokens,
    tokens_counts_indices: dict[str, list[int]],
    positions: dict[str, list[int]],
):
    new_tokens_counts: dict[str, int] = dict()
    new_positions: dict[str, list[int]] = dict()
    for rt, st in zip(raw_tokens, stemmed_tokens):
        if st not in new_positions:
            new_positions[st] = []
        new_positions[st] += positions[rt]
    for t in new_positions.keys():
        new_positions[t] = sorted(list(set(new_positions[t])))
        new_tokens_counts[t] = len(new_positions[t])
    return new_tokens_counts, new_positions


def process(dataset: dict[str, DatasetDocument]) -> Index:
    for v in dataset.values():
        tokens, tokens_counts_indices, tokens_positions = tokenize(normalize(v.content))
        stemmed_tokens = stem(tokens)
        tokens_counts, tokens_positions = merge_positional_indices(
            tokens, stemmed_tokens, tokens_counts_indices, tokens_positions
        )
        for token in tokens_counts.keys():
            count, positions = tokens_counts[token], tokens_positions[token]
            v.tokens.append(DatasetDocumentToken(token, positions, count))
    index = index_dataset(dataset)
    return index


def answer_query(index: Index, query: str):
    pass


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
        index = process(dataset)
        print(f"processed dataset in {time.time() - start}s")
        print(repr(dataset))
        exit(0)

    if args.save_index:
        with open(args.save_index, "wb") as save_file:
            pickle.dump(index, save_file, 0)

    if args.query:
        answer_query(index, args.query)

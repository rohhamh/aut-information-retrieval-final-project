import re
from hazm import Normalizer, Stemmer

normalizer = Normalizer(
    # correct_spacing breaks emails, it's initially avoided and will be applied again during tokenization
    correct_spacing=False,
)
correct_spacing = Normalizer().correct_spacing

def get_string_tokens(text: str):
    normalized_text = normalize(text)
    tokens, tokens_positions = tokenize(normalized_text)
    stemmed_tokens = stem(tokens)
    tokens_counts, tokens_positions = merge_positional_indices(
        tokens, stemmed_tokens, tokens_positions
    )
    return tokens_counts, tokens_positions


def normalize(text: str):
    return normalizer.normalize(text)


def tokenize(text: str):
    delimiters_pattern = re.compile(r'([؟!?]+|[\d.:]+|[:.،؛»\])}"«\[({/\\])')  # TODO \d
    id_pattern = re.compile(r"(?<![\w._])(@[\w_]+)")
    link_pattern = re.compile(
        r"((https?|ftp)://)?(?<!@)(([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})[-\w@:%_.+/~#?=&]*",
    )
    email_pattern = re.compile(
        r"[a-zA-Z0-9._+-]+@([a-zA-Z0-9-]+\.)+[A-Za-z]{2,}",
    )
    number_int_pattern = re.compile(
        r"\b(?<![\d۰-۹][.٫٬,])([\d۰-۹]+)(?![.٫٬,][\d۰-۹])\b",
    )
    number_float_pattern = re.compile(
        r"\b(?<!\.)([\d۰-۹,٬]+[.٫٬][\d۰-۹]+)\b(?!\.)",
    )
    patterns = [
        link_pattern,
        email_pattern,
        id_pattern,
        number_int_pattern,
        number_float_pattern,
    ]
    combined_patterns = "(" + ")|(".join(map(lambda p: p.pattern, patterns)) + ")"
    new_text_last_idx = 0
    new_text = ""
    text = text.replace("\n", " ").replace("\t", " ")
    positions: dict[str, list[int]] = {}
    for match in re.finditer(r"\S+", text):
        match_start, match_end = match.start(), match.end()
        new_text += text[new_text_last_idx:match_start]
        group = match.group(0)
        applied_delimiter_pattern = False
        for m in re.finditer(combined_patterns, group):
            if m and m.end() < len(group):
                group = (
                    group[m.start() : m.end()]
                    + " "
                    + delimiters_pattern.sub(r" \1 ", correct_spacing(group[m.end() :]))
                )
                applied_delimiter_pattern = True
        if not applied_delimiter_pattern:
            group = correct_spacing(group)
            group = delimiters_pattern.sub(r" \1 ", group)
        new_text += f" {group} "
        new_text_last_idx = match_end
        for t in group.split(" "):
            if t not in positions:
                positions[t] = []
            positions[t].append(match_start)
    text = new_text

    tokens = [word for word in text.split(" ") if word]

    tokens = list(filter(lambda t: not re.match(r"(EMAIL)|(NUM\d*)|(LINK)", t), tokens))
    tokens = list(filter(lambda t: not re.match(r"\b\d+\b", t), tokens))
    return tokens, positions


def stem(terms: list[str]):
    stemmer = Stemmer()
    return list(map(stemmer.stem, terms))


def merge_positional_indices(
    raw_tokens,
    stemmed_tokens,
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

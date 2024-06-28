class Term:
    def __init__(self, term: str, idf: int, postings_list) -> None:
        self.term = term
        self.idf = idf
        self.postings = postings_list

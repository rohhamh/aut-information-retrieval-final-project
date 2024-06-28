class Posting:
    def __init__(
        self, document_id: str, term_freq: int, term_positions: list[int]
    ) -> None:
        self.document_id = document_id
        self.term_freq = term_freq
        self.term_positions: list[int] = term_positions

    def __repr__(self) -> str:
        return str(self.__dict__)


class IndexItem:
    def __init__(self, df: int, tf: int, postings_list: list[Posting], champtions_list: list[Posting]) -> None:
        self.df = df
        self.tf = tf
        self.postings = postings_list
        self.champtions_list: list[Posting] = champtions_list

    def __repr__(self) -> str:
        return str(self.__dict__)


class Index:
    def __init__(self, dataset_hash: bytes) -> None:
        self._index: dict[str, IndexItem] = dict()
        self.dataset_hash = dataset_hash
        self.document_vectors: dict[str, list[tuple[str, float]]] = {}

    def set(self, k: str, ii: IndexItem):
        self._index[k] = ii
        self._index[k].df += len(ii.postings)

    def add(self, k: str, p: Posting):
        if k not in self._index:
            self._index[k] = IndexItem(1, p.term_freq, [p], [])
            return
        self._index[k].df += 1
        self._index[k].tf += p.term_freq
        self._index[k].postings.append(p)

    def has(self, k: str):
        return k in self._index

    def get(self, k: str) -> IndexItem | None:
        if k not in self._index:
            return None
        return self._index[k]

    def items(self):
        return self._index.items()

    def values(self):
        return self._index.values()

    def remove(self, k: str):
        del self._index[k]

    def __repr__(self) -> str:
        return str(self.__dict__)


class DatasetDocument:
    def __init__(
        self,
        id: str,
        title: str,
        content: str,
        tags: list[str],
        date: str,
        url: str,
        category: str,
    ) -> None:
        self.id = id
        self.title = title
        self.content = content
        self.tags = tags
        self.date = date
        self.url = url
        self.tokens: list[DatasetDocumentToken] = []
        self.category = category
        self.vector: list[tuple[str, float]]

    def __repr__(self) -> str:
        return str(self.__dict__)


class QueryResult:
    def __init__(self, id: str, title: str, url: str) -> None:
        self.id = id
        self.title = title
        self.url = url

    def __repr__(self) -> str:
        return str(self.__dict__)


class DatasetDocumentToken:
    def __init__(
        self, token: str, position_indices: list[int], token_freq: int
    ) -> None:
        self.token = token
        self.position_indices = position_indices
        self.token_freq = token_freq

    def __repr__(self) -> str:
        return str(self.__dict__)

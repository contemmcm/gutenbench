import glob
import json
import os

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, ClassVar
from abc import ABC, abstractmethod
from tqdm import tqdm

import torch

from accelerate.utils import find_executable_batch_size

from transformers import AutoModel, AutoTokenizer

from gutenbench.settings import INDEXES_DIR
from gutenbench.utils import load_docids, write_trec_run


def _iter_input_files(input_path: str) -> Iterable[str]:
    if os.path.isfile(input_path):
        return [input_path]

    patterns = ["*.json", "*.jsonl"]
    files: List[str] = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(os.path.join(input_path, pattern))))
    return files


def _read_json_obj(path: str) -> Iterable[Dict]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for obj in data:
                yield obj
        else:
            yield data


@dataclass
class BaseConfig:
    device: str = None


class BaseModel(ABC):

    registry: ClassVar[dict[str, type["BaseModel"]]] = {}
    name: ClassVar[str | None] = None
    config_class: ClassVar[type | None] = None
    query_encoder_class: ClassVar[type | None] = None

    def __init__(self, config: BaseConfig):
        self.config = config
        self.query_encoder = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name is not None:
            key = cls.name
            if key in BaseModel.registry:
                raise ValueError(f"Duplicate encoder registration: {key}")
            if cls.config_class is None:
                raise ValueError(f"{cls.__name__} must define config_class")
            BaseModel.registry[key] = cls

    @abstractmethod
    def tokenize(self, input_texts: list[str], **kwargs):
        pass

    @abstractmethod
    def make_searcher(self, corpus: str):
        pass

    @staticmethod
    def _resolve_device(device_arg: str | None) -> torch.device:
        if device_arg:
            return torch.device(device_arg)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def index_name(self) -> str:
        return self.name

    @staticmethod
    def _index_path(corpus: str, model: str) -> str:
        return str(INDEXES_DIR / corpus / model)

    def format_documents(self, documents: list[str]) -> list[str]:
        """
        Format documents for encoding. By default, this is a no-op, but can be
        overridden by subclasses if needed.
        """
        return documents

    def format_queries(self, queries: list[str]) -> list[str]:
        """
        Format queries for encoding. By default, this is a no-op, but can be
        overridden by subclasses if needed.
        """
        return queries

    def get_searcher(self, corpus: str):
        """
        Get a FaissSearcher for the specified corpus, creating it if it doesn't exist.
        """
        if corpus in self.query_encoder:
            return self.query_encoder[corpus]

        self.query_encoder[corpus] = self.make_searcher(corpus)

        return self.query_encoder[corpus]

    def search(self, query: str, corpus: str, k: int = 10):
        """
        Search for relevant passages in the specified corpus.
        """
        return self.get_searcher(corpus).search(query, k)

    def load_documents(
        self, input_path: str, field: str = "contents"
    ) -> List[Tuple[str, str]]:
        """
        Load documents from a directory of JSON files.

        Each file must contain a document ID and a text field, for example:

            {
                "id": "doc1",
                "contents": "this is the contents."
            }

        Example directory:

            path/to/docs/
             - doc1.json
             - doc2.json
             - doc3.json

        Args:
            input_path: Path to the directory containing the JSON files.
            field: Name of the JSON field that contains the document text.
                Defaults to "contents".

        Returns:
            A list of (document_id, text) tuples, for example:

                [
                    ("doc1", "this is the contents."),
                    ("doc2", "this is the contents."),
                    ("doc3", "this is the contents."),
                ]
        """
        docs: List[Tuple[str, str]] = []
        files = list(_iter_input_files(input_path))
        if not files:
            raise FileNotFoundError(f"No .json/.jsonl files found at: {input_path}")

        for path in files:
            for obj in _read_json_obj(path):
                doc_id = obj.get("id", obj.get("_id", obj.get("docid")))
                if doc_id is None:
                    raise ValueError(f"Missing id/_id/docid in file {path}")

                if field in obj:
                    text = obj[field]
                elif field == "contents" and "text" in obj:
                    text = obj["text"]
                else:
                    raise ValueError(
                        f"Missing field '{field}' in document {doc_id} from {path}"
                    )

                if isinstance(text, list):
                    text = "\n".join(str(x) for x in text)
                else:
                    text = str(text)

                docs.append((str(doc_id), text))

        return docs

    def _inner_save_doc_embeddings(
        self, batch_size: int, docs: List[Tuple[str, str]], output: str
    ):
        """
        Inner encoding loop. Takes batch_size as its first argument so that
        find_executable_batch_size can inject and retry it on OOM.
        """
        print(f"Encoding with batch_size={batch_size}")
        os.makedirs(output, exist_ok=True)
        out_path = os.path.join(output, "embeddings.jsonl")

        with open(out_path, "w", encoding="utf-8") as out_f:
            for start in tqdm(range(0, len(docs), batch_size), desc="Encoding"):
                batch = docs[start : start + batch_size]

                texts = self.format_documents([text for _, text in batch])
                embeddings = self.tokenize(input_texts=texts)

                for (doc_id, original_text), vec in zip(batch, embeddings):
                    out_f.write(
                        json.dumps(
                            {
                                "id": doc_id,
                                "contents": original_text,
                                "vector": vec.tolist(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        print(f"Wrote Pyserini-compatible embeddings to: {out_path}")

    def save_doc_embeddings(
        self,
        input: str,
        field: str,
        output: str,
        batch_size: int,
        auto_batch_size: bool = False,
    ):
        """
        Encode all documents from the corpus and save the embeddings to disk in a
        Pyserini-compatible format.

        When auto_batch_size=True, batch_size is used as the starting point and
        halved on each OOM, mirroring the HuggingFace Trainer's auto_find_batch_size
        behaviour.
        """
        print(f"Loading corpus from {input}")
        docs = self.load_documents(input, field)
        print(f"Loaded {len(docs)} documents")

        if auto_batch_size:
            state = [batch_size]

            def halve():
                state[0] //= 2
                return state[0]

            inner = find_executable_batch_size(
                self._inner_save_doc_embeddings, batch_size, halve
            )
            inner(docs=docs, output=output)
        else:
            self._inner_save_doc_embeddings(batch_size, docs=docs, output=output)

    def _inner_search_run(
        self,
        batch_size: int,
        index,
        docids: list[str],
        topics: list[tuple[str, str]],
        output_path: str,
        hits: int,
        tag: str,
        queries_cache_path: str = None,
    ):
        """
        Inner search loop. Takes batch_size as its first argument so that
        find_executable_batch_size can inject and retry it on OOM.
        """
        print(f"Searching with batch_size={batch_size}")
        results: list[tuple[str, list[tuple[str, float]]]] = []

        cache_file = None
        if queries_cache_path:
            os.makedirs(queries_cache_path, exist_ok=True)
            cache_file = open(os.path.join(queries_cache_path, "embeddings.jsonl"), "w", encoding="utf-8")

        try:
            for start in tqdm(range(0, len(topics), batch_size), desc="Searching"):
                batch = topics[start : start + batch_size]
                batch_ids = [topic_id for topic_id, _ in batch]
                batch_queries = self.format_queries([text for _, text in batch])

                query_vectors = self.tokenize(
                    input_texts=batch_queries,
                    max_length=self.config.max_length,
                )

                if cache_file is not None:
                    for query_id, vec in zip(batch_ids, query_vectors):
                        cache_file.write(json.dumps({"id": query_id, "vector": vec.tolist()}) + "\n")

                scores, indexes = index.search(query_vectors, hits)

                for query_id, row_scores, row_indexes in zip(batch_ids, scores, indexes):
                    result_hits = [
                        (docids[doc_idx], float(score))
                        for score, doc_idx in zip(row_scores, row_indexes)
                        if doc_idx != -1
                    ]
                    results.append((query_id, result_hits))
        finally:
            if cache_file is not None:
                cache_file.close()
                print(f"Wrote query embeddings to: {cache_file.name}")

        write_trec_run(output_path, results, tag)

    def search_run(
        self,
        index_path: str,
        topics_path: str,
        output_path: str,
        batch_size: int,
        hits: int = 1000,
        tag: str = "QWEN",
        threads: int = 1,
        auto_batch_size: bool = False,
        queries_cache_path: str = None,
    ):
        """
        Run a search and save the results in TREC format.

        When auto_batch_size=True, batch_size is used as the starting point and
        halved on each OOM.
        """
        import faiss
        from pyserini.query_iterator import DefaultQueryIterator

        faiss.omp_set_num_threads(threads)
        index = faiss.read_index(f"{index_path}/index")
        docids = load_docids(f"{index_path}/docid")
        topics = list(
            (str(topic_id), text)
            for topic_id, text in DefaultQueryIterator.from_topics(topics_path)
        )

        kwargs = dict(index=index, docids=docids, topics=topics, output_path=output_path, hits=hits, tag=tag, queries_cache_path=queries_cache_path)

        if auto_batch_size:
            state = [batch_size]

            def halve():
                state[0] //= 2
                return state[0]

            inner = find_executable_batch_size(self._inner_search_run, batch_size, halve)
            inner(**kwargs)
        else:
            self._inner_search_run(batch_size, **kwargs)


class TransfomerModel(BaseModel):

    _model: AutoModel = None
    _tokenizer: AutoTokenizer = None

    @abstractmethod
    def make_tokenizer(self):
        pass

    @abstractmethod
    def make_model(self):
        pass

    @property
    def model(self):
        """
        Lazy load the model to avoid unnecessary memory usage when only searching.
        """
        if not self._model:
            self._model = self.make_model()
            self._model.to(self._resolve_device(self.config.device))
            self._model.eval()
        return self._model

    @property
    def tokenizer(self):
        """
        Lazy load the tokenizer to avoid unnecessary memory usage when only searching.
        """
        if not self._tokenizer:
            self._tokenizer = self.make_tokenizer()
        return self._tokenizer

    def make_searcher(self, corpus: str):
        """
        Instantiate a FaissSearcher for the specified corpus.
        """
        from pyserini.search.faiss import FaissSearcher

        return FaissSearcher(
            self._index_path(corpus, self.index_name), self.query_encoder_class(self)
        )


class SparseModel(BaseModel):
    """
    Base class for learned sparse retrieval models (e.g. DeepCT).

    Instead of producing dense vectors, these models predict per-token importance
    weights and output weighted term-frequency text for Lucene/BM25 indexing.
    Search is done via pyserini.search.lucene, not FAISS.
    """

    _model: AutoModel = None
    _tokenizer: AutoTokenizer = None

    @abstractmethod
    def make_model(self):
        pass

    @abstractmethod
    def make_tokenizer(self):
        pass

    @property
    def model(self):
        if not self._model:
            self._model = self.make_model()
            self._model.to(self._resolve_device(self.config.device))
            self._model.eval()
        return self._model

    @property
    def tokenizer(self):
        if not self._tokenizer:
            self._tokenizer = self.make_tokenizer()
        return self._tokenizer

    def make_searcher(self, _corpus: str):
        raise NotImplementedError(
            "Sparse models use Lucene; run pyserini.search.lucene directly"
        )

    def _inner_save_doc_embeddings(
        self, batch_size: int, docs: List[Tuple[str, str]], output: str
    ):
        """
        Inner encoding loop for sparse models. Outputs weighted-term text for
        Lucene indexing instead of dense vectors.
        tokenize() must return a list[str] of weighted-text strings.
        """
        print(f"Encoding with batch_size={batch_size}")
        os.makedirs(output, exist_ok=True)
        out_path = os.path.join(output, "docs.jsonl")

        with open(out_path, "w", encoding="utf-8") as out_f:
            for start in tqdm(range(0, len(docs), batch_size), desc="Encoding"):
                batch = docs[start : start + batch_size]
                texts = self.format_documents([text for _, text in batch])
                weighted_texts = self.tokenize(input_texts=texts)

                for (doc_id, _), weighted_text in zip(batch, weighted_texts):
                    out_f.write(
                        json.dumps(
                            {"id": doc_id, "contents": weighted_text},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        print(f"Wrote weighted-term documents to: {out_path}")


class SentenceTransformerModel(BaseModel):

    _model: AutoModel = None

    @abstractmethod
    def make_model(self):
        pass

    @property
    def model(self):
        """
        Lazy load the model to avoid unnecessary memory usage when only searching.
        """
        if not self._model:
            self._model = self.make_model()
            self._model.to(self._resolve_device(self.config.device))
            self._model.eval()
        return self._model

    def make_searcher(self, corpus: str):
        """
        Instantiate a FaissSearcher for the specified corpus.
        """
        from pyserini.search.faiss import FaissSearcher

        return FaissSearcher(
            self._index_path(corpus, self.index_name), self.query_encoder_class(self)
        )

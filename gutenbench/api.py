"""
A small Flask API exposing search over corpora that have already been indexed
via the Makefile (see README.md: `make QWEN/KJV`, `make BM25/KJV`, etc.).

Run with:

    python -m gutenbench.api

Example request:

    curl -X POST http://localhost:8008/search \\
        -H 'Content-Type: application/json' \\
        -d '{
            "query": "For how long did the flood rain fall?",
            "corpus": "KJV",
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "top_k": 10
        }'
"""

import threading
from collections import namedtuple
from dataclasses import fields

from flask import Flask, jsonify, request

from gutenbench import models
from gutenbench.models.base import SparseModel
from gutenbench.settings import INDEXES_DIR

BM25 = "BM25"

# Result shape shared between BM25's pyserini ScoredDoc and our own manually
# assembled dense hits below, so both can be formatted identically.
Hit = namedtuple("Hit", ["docid", "score"])

app = Flask(__name__)

_cache_lock = threading.Lock()
_bm25_searcher_cache = {}

# Only one dense encoder's weights are kept resident at a time (many of these
# models barely fit a single GPU), so switching `model` unloads the previous
# one. Repeated requests for the *same* model reuse it without a reload.
_active_model = {"key": None, "instance": None}
_dense_searcher_cache = {}


class SearchError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _build_model_id_index():
    """
    Map every HF model id a registered dense encoder can be configured with
    (its config class' `model_id` choices) to that encoder's registry name,
    e.g. "Qwen/Qwen3-Embedding-0.6B" -> "Qwen3".
    """
    index = {}
    for name, cls in models.BaseModel.registry.items():
        if issubclass(cls, SparseModel):
            continue
        for f in fields(cls.config_class):
            if f.name != "model_id":
                continue
            for choice in f.metadata.get("choices", [f.default]):
                index[choice] = name
    return index


MODEL_ID_TO_ENCODER = _build_model_id_index()

# Registry name -> Makefile target prefix, for the rare cases where they diverge
# (e.g. encoder "Qwen3" is built via `make QWEN/<corpus>`, not `make QWEN3/<corpus>`).
MAKE_TARGET = {"Qwen3": "QWEN"}


def _resolve_index_dir(corpus: str, encoder_name: str, model_id: str):
    """
    Index directories are named after the Makefile's REPORT_DIR convention
    (namespace__version), except for single-version families whose directory
    is just the encoder name (see models/BGE.mk).
    """
    for candidate in (
        INDEXES_DIR / corpus / model_id.replace("/", "__"),
        INDEXES_DIR / corpus / encoder_name,
    ):
        if candidate.is_dir():
            return candidate

    target = MAKE_TARGET.get(encoder_name, encoder_name.upper())
    raise SearchError(
        f"No index found for model {model_id!r} on corpus {corpus!r}. "
        f"Build it first, e.g.: make {target}/{corpus}",
        404,
    )


def _get_bm25_searcher(corpus: str):
    cache_key = (corpus, BM25)
    if cache_key in _bm25_searcher_cache:
        return _bm25_searcher_cache[cache_key]

    with _cache_lock:
        if cache_key in _bm25_searcher_cache:
            return _bm25_searcher_cache[cache_key]

        index_dir = INDEXES_DIR / corpus / BM25
        if not index_dir.is_dir():
            raise SearchError(
                f"No BM25 index found for corpus {corpus!r}. "
                f"Build it first: make BM25/{corpus}",
                404,
            )

        from pyserini.search.lucene import LuceneSearcher

        searcher = LuceneSearcher(str(index_dir))
        _bm25_searcher_cache[cache_key] = searcher
        return searcher


def _unload_active_model():
    """
    Drop the resident dense encoder (if any) and free its GPU memory. Also
    drops cached FaissSearchers, since they hold a reference to it via their
    query encoder.
    """
    if _active_model["instance"] is None:
        return

    print(f"Unloading model {_active_model['key']!r} to free GPU memory")
    _active_model["instance"] = None
    _active_model["key"] = None
    _dense_searcher_cache.clear()

    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_active_model(encoder_name: str, model_id: str):
    key = (encoder_name, model_id)

    with _cache_lock:
        if _active_model["key"] == key:
            return _active_model["instance"]

        _unload_active_model()

        model_cls = models.get_model_class(encoder_name)
        config_cls = models.get_config_class(encoder_name)
        model = model_cls(config_cls(model_id=model_id))

        _active_model["key"] = key
        _active_model["instance"] = model
        return model


def _dense_model_and_searcher(corpus: str, model_id: str):
    encoder_name = MODEL_ID_TO_ENCODER.get(model_id)
    if encoder_name is None:
        raise SearchError(
            f"Unknown model: {model_id!r}. Known models: "
            f"{sorted(MODEL_ID_TO_ENCODER)}",
            400,
        )

    index_dir = _resolve_index_dir(corpus, encoder_name, model_id)

    # Loading/unloading the model must happen before the cache_key lookup
    # below: a cached searcher from a since-unloaded model must not be reused.
    model = _get_active_model(encoder_name, model_id)

    cache_key = (corpus, str(index_dir))
    if cache_key in _dense_searcher_cache:
        return model, _dense_searcher_cache[cache_key]

    with _cache_lock:
        if cache_key in _dense_searcher_cache:
            return model, _dense_searcher_cache[cache_key]

        from pyserini.search.faiss import FaissSearcher

        searcher = FaissSearcher(str(index_dir), model.query_encoder_class(model))
        _dense_searcher_cache[cache_key] = searcher
        return model, searcher


def _search_hits_batch(corpus: str, model_id: str, queries: list[str], top_k: int):
    """
    Search a batch of queries at once. For dense models this runs a single
    batched forward pass through the encoder instead of one per query, which
    is where most of the speedup over sequential /search calls comes from.
    """
    if model_id == BM25:
        searcher = _get_bm25_searcher(corpus)
        qids = [str(i) for i in range(len(queries))]
        hits_by_qid = searcher.batch_search(queries, qids, k=top_k)
        return [hits_by_qid.get(qid, []) for qid in qids]

    model, searcher = _dense_model_and_searcher(corpus, model_id)

    formatted = model.format_queries(list(queries))
    query_vectors = model.tokenize(formatted, max_length=model.config.max_length)
    distances, indexes = searcher.index.search(query_vectors, top_k)

    return [
        [
            Hit(searcher.docids[idx], float(score))
            for score, idx in zip(row_scores, row_indexes)
            if idx != -1
        ]
        for row_scores, row_indexes in zip(distances, indexes)
    ]


def _parse_search_request(payload: dict):
    query = payload.get("query")
    corpus = payload.get("corpus")
    model_id = payload.get("model")
    top_k = payload.get("top_k", 10)

    is_batch = isinstance(query, list)
    queries = query if is_batch else [query]

    if not queries or not all(isinstance(q, str) and q.strip() for q in queries):
        raise SearchError(
            "'query' is required and must be a non-empty string, or a "
            "non-empty list of non-empty strings",
            400,
        )
    if not isinstance(corpus, str) or not corpus.strip():
        raise SearchError("'corpus' is required and must be a non-empty string", 400)
    if not isinstance(model_id, str) or not model_id.strip():
        raise SearchError("'model' is required and must be a non-empty string", 400)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise SearchError("'top_k' must be a positive integer", 400)

    return queries, is_batch, corpus, model_id, top_k


@app.post("/search")
def search():
    payload = request.get_json(silent=True) or {}

    try:
        queries, is_batch, corpus, model_id, top_k = _parse_search_request(payload)
        hits_per_query = _search_hits_batch(corpus, model_id, queries, top_k)
    except SearchError as exc:
        return jsonify(error=exc.message), exc.status_code

    results_per_query = [
        [
            {"rank": rank, "doc_id": hit.docid, "score": float(hit.score)}
            for rank, hit in enumerate(hits, start=1)
        ]
        for hits in hits_per_query
    ]

    if is_batch:
        return jsonify(
            queries=queries, corpus=corpus, model=model_id, results=results_per_query
        )

    return jsonify(
        query=queries[0], corpus=corpus, model=model_id, results=results_per_query[0]
    )


def main():
    app.run(host="0.0.0.0", port=8008)


if __name__ == "__main__":
    main()

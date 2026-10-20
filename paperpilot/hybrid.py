"""Hybrid BM25 + vector retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from paperpilot.store import PaperStore

logger = logging.getLogger(__name__)

# Cache: (frozenset(paper_ids), chunk_count) -> (BM25Okapi, id_list, chunk_by_id)
_bm25_cache: dict[
    tuple[frozenset[str], int],
    tuple[BM25Okapi, list[str], dict[str, dict[str, Any]]],
] = {}


def rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: score ids by ``1/(k + rank)`` across lists; highest first."""
    scores: dict[str, float] = {}
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)


def _tokenize(text: str) -> list[str]:
    """Lowercased whitespace tokens."""
    return text.lower().split()


def _bm25_index(
    store: PaperStore,
    paper_ids: list[str] | None,
) -> tuple[BM25Okapi, list[str], dict[str, dict[str, Any]]]:
    """Return a BM25 index for the scoped chunks, rebuilding when the library changes."""
    chunks = store.get_chunks(paper_ids)
    key = (frozenset(paper_ids or []), len(chunks))
    cached = _bm25_cache.get(key)
    if cached is not None:
        return cached

    ids = [c["id"] for c in chunks]
    corpus = [_tokenize(c.get("text") or "") for c in chunks]
    bm25 = BM25Okapi(corpus) if corpus else BM25Okapi([[""]])
    chunk_by_id = {c["id"]: c for c in chunks}
    _bm25_cache[key] = (bm25, ids, chunk_by_id)
    logger.info("Built BM25 index chunks=%d scoped=%s", len(ids), bool(paper_ids))
    return _bm25_cache[key]


def hybrid_search(
    store: PaperStore,
    query: str,
    paper_ids: list[str] | None = None,
    k: int = 6,
    min_score: float = 0.30,
) -> list[dict[str, Any]]:
    """Fuse BM25 and vector top-20 lists with RRF; return top-``k`` hit dicts."""
    bm25, ids, chunk_by_id = _bm25_index(store, paper_ids)
    if not ids:
        return []

    query_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_ranked = [
        doc_id
        for doc_id, _score in sorted(
            zip(ids, bm25_scores),
            key=lambda item: item[1],
            reverse=True,
        )[:20]
    ]

    vector_hits = store.search(query, k=20, paper_ids=paper_ids)
    vector_hits = [h for h in vector_hits if h["score"] >= min_score]
    vector_ids = [h["id"] for h in vector_hits]
    vector_score_by_id = {h["id"]: float(h["score"]) for h in vector_hits}

    fused_ids = rrf([bm25_ranked, vector_ids])
    results: list[dict[str, Any]] = []
    for doc_id in fused_ids[:k]:
        chunk = chunk_by_id.get(doc_id)
        if chunk is None:
            continue
        results.append(
            {
                "id": doc_id,
                "text": chunk.get("text", ""),
                "meta": chunk.get("meta") or {},
                "score": vector_score_by_id.get(doc_id, 0.0),
            }
        )
    return results

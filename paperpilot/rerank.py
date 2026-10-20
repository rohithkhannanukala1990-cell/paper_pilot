"""Cross-encoder reranking over retrieved hits."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_RERANKER_MODEL = "BAAI/bge-reranker-base"
_cross_encoder: Any | None = None


def _get_cross_encoder() -> Any:
    """Lazy-load the shared CrossEncoder singleton."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder %s", _RERANKER_MODEL)
        _cross_encoder = CrossEncoder(_RERANKER_MODEL)
    return _cross_encoder


def rerank(query: str, hits: list[dict[str, Any]], top_n: int = 5) -> list[dict[str, Any]]:
    """Score ``(query, hit text)`` pairs; return top_n sorted by cross-encoder score."""
    if not hits:
        return []
    t0 = time.perf_counter()
    pairs = [(query, hit.get("text") or "") for hit in hits]
    scores = _get_cross_encoder().predict(pairs)
    ranked: list[dict[str, Any]] = []
    for hit, ce_score in zip(hits, scores):
        item = dict(hit)
        item["rerank_score"] = float(ce_score)
        ranked.append(item)
    ranked.sort(key=lambda h: h["rerank_score"], reverse=True)
    logger.info(
        "rerank n_in=%d top_n=%d latency=%.2fs",
        len(hits),
        top_n,
        time.perf_counter() - t0,
    )
    return ranked[:top_n]

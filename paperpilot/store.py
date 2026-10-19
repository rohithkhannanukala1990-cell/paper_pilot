"""ChromaDB paper store with BGE sentence-transformer embeddings."""

from __future__ import annotations

import logging
import time
from typing import Any

import chromadb

from paperpilot.config import settings
from paperpilot.ingest import Chunk

logger = logging.getLogger(__name__)

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model: Any | None = None


def _get_model() -> Any:
    """Lazy-load the shared SentenceTransformer singleton."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s", settings.embed_model)
        _model = SentenceTransformer(settings.embed_model)
    return _model


def embed_docs(texts: list[str]) -> list[list[float]]:
    """Embed documents with L2-normalized vectors (no query prefix)."""
    if not texts:
        return []
    t0 = time.perf_counter()
    vectors = _get_model().encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    logger.info("embed docs n=%d latency=%.2fs", len(texts), time.perf_counter() - t0)
    return vectors.tolist()


def embed_query(q: str) -> list[float]:
    """Embed a query with the BGE query prefix and L2 normalization."""
    t0 = time.perf_counter()
    vector = _get_model().encode(
        QUERY_PREFIX + q,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    logger.info("embed query latency=%.2fs", time.perf_counter() - t0)
    return vector.tolist()


def _where(paper_ids: list[str] | None) -> dict[str, Any] | None:
    """Build a Chroma where-clause; never emit an empty ``$in`` list."""
    if not paper_ids:
        return None
    return {"paper_id": {"$in": paper_ids}}


class PaperStore:
    """Persistent Chroma collection of paper chunks."""

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=settings.chroma_dir)
        self._col = self._client.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"},
        )

    def has_paper(self, paper_id: str) -> bool:
        """Return True if any chunk exists for ``paper_id``."""
        result = self._col.get(where={"paper_id": paper_id}, limit=1)
        return bool(result["ids"])

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Upsert chunks keyed by ``chunk_id``."""
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "paper_id": c.paper_id,
                "title": c.title,
                "section": c.section,
                "page": c.page,
            }
            for c in chunks
        ]
        embeddings = embed_docs(documents)
        self._col.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("Upserted %d chunks", len(chunks))

    def search(
        self,
        query: str,
        k: int = 6,
        paper_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-``k`` hits as ``{id, text, meta, score}`` (score = 1 - distance)."""
        t0 = time.perf_counter()
        where = _where(paper_ids)
        kwargs: dict[str, Any] = {
            "query_embeddings": [embed_query(query)],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where
        result = self._col.query(**kwargs)

        hits: list[dict[str, Any]] = []
        ids = result["ids"][0] if result["ids"] else []
        docs = result["documents"][0] if result["documents"] else []
        metas = result["metadatas"][0] if result["metadatas"] else []
        dists = result["distances"][0] if result["distances"] else []
        for i, doc_id in enumerate(ids):
            hits.append(
                {
                    "id": doc_id,
                    "text": docs[i],
                    "meta": metas[i],
                    "score": 1.0 - float(dists[i]),
                }
            )
        logger.info(
            "retrieve k=%d hits=%d scoped=%s latency=%.2fs",
            k,
            len(hits),
            bool(paper_ids),
            time.perf_counter() - t0,
        )
        return hits

    def get_chunks(self, paper_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Return stored chunks as ``{id, text, meta}``, optionally filtered."""
        where = _where(paper_ids)
        kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        if where is not None:
            kwargs["where"] = where
        result = self._col.get(**kwargs)
        chunks: list[dict[str, Any]] = []
        for i, doc_id in enumerate(result["ids"]):
            chunks.append(
                {
                    "id": doc_id,
                    "text": result["documents"][i] if result["documents"] else "",
                    "meta": result["metadatas"][i] if result["metadatas"] else {},
                }
            )
        return chunks

    def list_papers(self) -> list[dict[str, Any]]:
        """Return ``[{paper_id, title, chunks}, ...]`` aggregated from metadata."""
        by_id: dict[str, dict[str, Any]] = {}
        for item in self.get_chunks():
            meta = item["meta"] or {}
            pid = meta.get("paper_id", "")
            if not pid:
                continue
            if pid not in by_id:
                by_id[pid] = {
                    "paper_id": pid,
                    "title": meta.get("title", ""),
                    "chunks": 0,
                }
            by_id[pid]["chunks"] += 1
        return list(by_id.values())

    def delete_paper(self, paper_id: str) -> None:
        """Delete all chunks for ``paper_id``."""
        self._col.delete(where={"paper_id": paper_id})
        logger.info("Deleted paper %s", paper_id)

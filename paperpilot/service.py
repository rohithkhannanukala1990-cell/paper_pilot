"""PaperPilot orchestration: ingest, retrieve, and answer with citations."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pymupdf as fitz

from paperpilot.config import MAX_PAGES, MAX_UPLOAD_MB, settings
from paperpilot.hybrid import hybrid_search
from paperpilot.ingest import ingest_pdf
from paperpilot.llm import complete, stream
from paperpilot.logging_setup import setup_logging
from paperpilot.prompts import (
    COMPARE_SYSTEM,
    FACETS,
    NOT_FOUND,
    SYSTEM,
    format_context,
    parse_citations,
)
from paperpilot.rerank import rerank
from paperpilot.store import PaperStore

logger = logging.getLogger(__name__)

MIN_SCORE = 0.30

REWRITE_SYSTEM = (
    "Rewrite the user's last question as a standalone search query, "
    "resolving pronouns using the conversation. Output only the query."
)


def standalone(history: list[tuple[str, str]] | None, question: str) -> str:
    """Return ``question``, or a rewritten standalone query when history is present."""
    if not history:
        return question
    turns = history[-6:]
    lines = [f"{role}: {text}" for role, text in turns]
    lines.append(f"user: {question}")
    t0 = time.perf_counter()
    rewritten = complete(REWRITE_SYSTEM, "\n".join(lines), max_tokens=80).strip()
    logger.info("generate rewrite latency=%.2fs", time.perf_counter() - t0)
    return rewritten


def _check_upload_limits(path: Path) -> None:
    """Raise ``ValueError`` when file size or page count exceeds configured limits."""
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise ValueError(
            f"This PDF is {size_mb:.1f} MB, which exceeds the "
            f"{MAX_UPLOAD_MB} MB upload limit. Try a smaller file."
        )
    with fitz.open(path) as doc:
        pages = doc.page_count
    if pages > MAX_PAGES:
        raise ValueError(
            f"This PDF has {pages} pages, which exceeds the "
            f"{MAX_PAGES}-page limit. Try a shorter document."
        )


class PaperPilot:
    """High-level API for indexing PDFs and answering grounded questions."""

    def __init__(self) -> None:
        setup_logging()
        self.store = PaperStore()

    def _retrieve(
        self,
        query: str,
        k: int,
        paper_ids: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Dispatch retrieval on ``settings.retriever``."""
        if settings.retriever == "vector":
            return self.store.search(query, k=k, paper_ids=paper_ids)
        if settings.retriever == "hybrid":
            return hybrid_search(
                self.store,
                query,
                paper_ids=paper_ids,
                k=k,
                min_score=MIN_SCORE,
            )
        if settings.retriever == "rerank":
            candidates = hybrid_search(
                self.store,
                query,
                paper_ids=paper_ids,
                k=20,
                min_score=MIN_SCORE,
            )
            return rerank(query, candidates, top_n=5)
        raise ValueError(f"Unknown retriever: {settings.retriever}")

    def add_pdf(self, path: str | Path) -> dict[str, Any]:
        """Ingest a PDF into the store; skip upsert if the paper is already indexed."""
        path = Path(path)
        _check_upload_limits(path)

        t_parse = time.perf_counter()
        chunks = ingest_pdf(path)
        logger.info(
            "parse path=%s chunks=%d latency=%.2fs",
            path.name,
            len(chunks),
            time.perf_counter() - t_parse,
        )

        if not chunks:
            raise ValueError("No extractable text found. Is this a scanned PDF?")
        paper_id = chunks[0].paper_id
        title = chunks[0].title
        if not self.store.has_paper(paper_id):
            self.store.add_chunks(chunks)
        else:
            logger.info("Paper %s already indexed; skipping add_chunks", paper_id)
        return {"paper_id": paper_id, "title": title, "chunks": len(chunks)}

    def ask(
        self,
        question: str,
        paper_ids: list[str] | None = None,
        history: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Retrieve with a rewritten query; answer using the original question."""
        query = standalone(history, question)
        hits = self._retrieve(query, settings.top_k, paper_ids)
        # hybrid/rerank already drop vector hits below MIN_SCORE before fusion
        if settings.retriever not in ("hybrid", "rerank"):
            hits = [h for h in hits if h["score"] >= MIN_SCORE]

        if not hits:
            return {
                "answer": NOT_FOUND,
                "sources": [],
                "invalid_citations": [],
                "query": query,
            }

        context = format_context(hits)
        user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
        t0 = time.perf_counter()
        answer = complete(SYSTEM, user)
        logger.info("generate ask latency=%.2fs", time.perf_counter() - t0)
        _valid, invalid = parse_citations(answer, hits)
        return {
            "answer": answer,
            "sources": hits,
            "invalid_citations": invalid,
            "query": query,
        }

    def ask_stream(
        self,
        question: str,
        paper_ids: list[str] | None = None,
        history: list[tuple[str, str]] | None = None,
    ) -> tuple[list[dict[str, Any]], str, Iterator[str]]:
        """Rewrite and retrieve sync; return ``(hits, query, text generator)``."""
        query = standalone(history, question)
        hits = self._retrieve(query, settings.top_k, paper_ids)
        if settings.retriever not in ("hybrid", "rerank"):
            hits = [h for h in hits if h["score"] >= MIN_SCORE]

        if not hits:

            def _not_found() -> Iterator[str]:
                yield NOT_FOUND

            return [], query, _not_found()

        context = format_context(hits)
        user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
        return hits, query, stream(SYSTEM, user)

    def _summarize_facet(
        self,
        paper_id: str,
        name: str,
        question: str,
    ) -> tuple[str, str, list[dict[str, Any]]]:
        """Retrieve and answer one summary facet for ``paper_id``."""
        hits = self.store.search(question, k=4, paper_ids=[paper_id])
        if not hits:
            return name, NOT_FOUND, []
        context = format_context(hits)
        user = (
            f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\n"
            "Answer in 2–4 sentences."
        )
        t0 = time.perf_counter()
        text = complete(SYSTEM, user, max_tokens=500)
        logger.info("generate facet=%s latency=%.2fs", name, time.perf_counter() - t0)
        return name, text, hits

    def summarize(self, paper_id: str) -> dict[str, Any]:
        """Summarize a paper by facet (concurrent), then write a 3-sentence TL;DR."""
        facet_items = list(FACETS.items())

        with ThreadPoolExecutor(max_workers=5) as pool:
            sections = list(
                pool.map(
                    lambda item: self._summarize_facet(paper_id, item[0], item[1]),
                    facet_items,
                )
            )

        notes = "\n\n".join(f"{name}:\n{text}" for name, text, _hits in sections)
        t0 = time.perf_counter()
        tldr = complete(
            "Write a 3-sentence TL;DR of this paper from the section notes. "
            "Do not include citation numbers. Output only the three sentences.",
            notes,
            max_tokens=500,
        ).strip()
        logger.info("generate tldr latency=%.2fs", time.perf_counter() - t0)

        return {
            "tldr": tldr,
            "sections": [
                {"name": name, "text": text, "hits": hits}
                for name, text, hits in sections
            ],
        }

    # Per-paper retrieval: each aspect is searched separately for every paper so a
    # long paper cannot crowd other papers out of a shared top-k result set.
    def compare(
        self,
        paper_ids: list[str],
        aspects: list[str],
    ) -> dict[str, Any]:
        """Compare papers on ``aspects`` with per-paper retrieval and citation checks."""
        by_id: dict[str, dict[str, Any]] = {}
        for aspect in aspects:
            for pid in paper_ids:
                for hit in self.store.search(aspect, k=3, paper_ids=[pid]):
                    by_id[hit["id"]] = hit

        hits = list(by_id.values())
        if not hits:
            return {
                "answer": NOT_FOUND,
                "sources": [],
                "invalid_citations": [],
            }

        titles_by_id = {
            p["paper_id"]: p["title"] for p in self.store.list_papers()
        }
        titles = [titles_by_id.get(pid, pid) for pid in paper_ids]
        context = format_context(hits)
        user = (
            f"PAPERS:\n"
            + "\n".join(f"- {t}" for t in titles)
            + f"\n\nASPECTS:\n"
            + "\n".join(f"- {a}" for a in aspects)
            + f"\n\nCONTEXT:\n{context}\n\n"
            "Compare the papers on the listed aspects."
        )
        t0 = time.perf_counter()
        answer = complete(COMPARE_SYSTEM, user, max_tokens=1800)
        logger.info("generate compare latency=%.2fs", time.perf_counter() - t0)
        _valid, invalid = parse_citations(answer, hits)
        return {
            "answer": answer,
            "sources": hits,
            "invalid_citations": invalid,
        }

"""Shared fixtures for offline PaperPilot tests."""

from __future__ import annotations

from typing import Any

import pytest

from paperpilot.llm import LLMError


@pytest.fixture
def fake_hits() -> list[dict[str, Any]]:
    """Two–three retrieval hits with id, text, meta, and score."""
    return [
        {
            "id": "p1-0",
            "text": "Transformers use self-attention for sequence modeling.",
            "meta": {
                "paper_id": "p1",
                "title": "Attention Is All You Need",
                "section": "Introduction",
                "page": 2,
            },
            "score": 0.82,
        },
        {
            "id": "p1-1",
            "text": "We evaluate on WMT 2014 English-German translation.",
            "meta": {
                "paper_id": "p1",
                "title": "Attention Is All You Need",
                "section": "Experiments",
                "page": 8,
            },
            "score": 0.71,
        },
        {
            "id": "p2-0",
            "text": "BERT is pretrained with masked language modeling.",
            "meta": {
                "paper_id": "p2",
                "title": "BERT",
                "section": "Method",
                "page": 3,
            },
            "score": 0.65,
        },
    ]


class _FakeStore:
    """Minimal store stub used by FakePilot."""

    def list_papers(self) -> list[dict[str, Any]]:
        return [
            {"paper_id": "p1", "title": "Attention Is All You Need", "chunks": 12},
            {"paper_id": "p2", "title": "BERT", "chunks": 8},
        ]

    def delete_paper(self, paper_id: str) -> None:
        return None

    def search(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    def has_paper(self, paper_id: str) -> bool:
        return False

    def add_chunks(self, chunks: list[Any]) -> None:
        return None

    def get_chunks(self, paper_ids: list[str] | None = None) -> list[dict[str, Any]]:
        return []


class FakePilot:
    """PaperPilot stub with fixed return values (no I/O)."""

    def __init__(self) -> None:
        self.store = _FakeStore()

    def add_pdf(self, path: Any) -> dict[str, Any]:
        return {"paper_id": "p1", "title": "Attention Is All You Need", "chunks": 12}

    def ask(
        self,
        question: str,
        paper_ids: list[str] | None = None,
        history: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        if question.startswith("FAIL_LLM"):
            raise LLMError("simulated provider failure")
        return {
            "answer": "Fixed answer [1].",
            "sources": [],
            "invalid_citations": [],
            "query": question,
        }

    def summarize(self, paper_id: str) -> dict[str, Any]:
        return {
            "tldr": "Fixed TL;DR sentence one. Sentence two. Sentence three.",
            "sections": [
                {"name": "Method", "text": "Fixed method note.", "hits": []},
            ],
        }

    def compare(self, paper_ids: list[str], aspects: list[str]) -> dict[str, Any]:
        return {
            "answer": "| aspect | p1 | p2 |\n|---|---|---|",
            "sources": [],
            "invalid_citations": [],
        }


@pytest.fixture
def fake_pilot() -> FakePilot:
    """Fresh FakePilot instance."""
    return FakePilot()

"""Offline tests for PaperPilot.ask with monkeypatched store and LLM."""

from __future__ import annotations

from typing import Any

import pytest

from paperpilot.prompts import NOT_FOUND
from paperpilot.service import MIN_SCORE, PaperPilot


class _StubStore:
    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self._hits = hits

    def search(
        self,
        query: str,
        k: int = 6,
        paper_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return list(self._hits)


class _VectorSettings:
    top_k = 6
    retriever = "vector"


def test_ask_returns_not_found_when_all_scores_below_min(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    low = [
        {
            "id": "x-0",
            "text": "irrelevant",
            "meta": {"title": "T", "section": "S", "page": 1},
            "score": MIN_SCORE - 0.05,
        },
        {
            "id": "x-1",
            "text": "also low",
            "meta": {"title": "T", "section": "S", "page": 2},
            "score": 0.1,
        },
    ]
    called = {"complete": False}

    def _fake_complete(*_args: Any, **_kwargs: Any) -> str:
        called["complete"] = True
        return "should not run"

    monkeypatch.setattr("paperpilot.service.PaperStore", lambda: _StubStore(low))
    monkeypatch.setattr("paperpilot.service.complete", _fake_complete)
    monkeypatch.setattr("paperpilot.service.settings", _VectorSettings())

    pilot = PaperPilot()
    result = pilot.ask("What is the contribution?")
    assert result["answer"] == NOT_FOUND
    assert result["sources"] == []
    assert result["invalid_citations"] == []
    assert called["complete"] is False


def test_ask_sources_preserve_text(monkeypatch: pytest.MonkeyPatch) -> None:
    hits = [
        {
            "id": "p-0",
            "text": "Distinctive passage about self-attention.",
            "meta": {
                "paper_id": "p",
                "title": "Demo",
                "section": "Method",
                "page": 4,
            },
            "score": 0.91,
        }
    ]

    monkeypatch.setattr("paperpilot.service.PaperStore", lambda: _StubStore(hits))
    monkeypatch.setattr(
        "paperpilot.service.complete",
        lambda *_a, **_k: "The method uses self-attention [1].",
    )
    monkeypatch.setattr("paperpilot.service.settings", _VectorSettings())

    pilot = PaperPilot()
    result = pilot.ask("How does the method work?")
    assert result["answer"].startswith("The method")
    assert result["sources"]
    assert "text" in result["sources"][0]
    assert result["sources"][0]["text"] == "Distinctive passage about self-attention."

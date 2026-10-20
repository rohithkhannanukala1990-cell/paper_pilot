"""Offline tests for context formatting and citation parsing."""

from __future__ import annotations

from paperpilot.prompts import format_context, parse_citations


def test_parse_citations_splits_valid_and_invalid(fake_hits) -> None:
    text = "Claim [1] and [2][2] plus bad [9] and [0]."
    valid, invalid = parse_citations(text, fake_hits)
    assert valid == [1, 2]
    assert invalid == [0, 9]


def test_format_context_numbering_and_labels(fake_hits) -> None:
    ctx = format_context(fake_hits[:2])
    assert "[1] (Attention Is All You Need | Introduction | p.2)" in ctx
    assert "[2] (Attention Is All You Need | Experiments | p.8)" in ctx
    assert "Transformers use self-attention" in ctx
    assert ctx.index("[1]") < ctx.index("[2]")

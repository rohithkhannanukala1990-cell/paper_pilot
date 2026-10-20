"""Offline tests for Reciprocal Rank Fusion."""

from __future__ import annotations

from paperpilot.hybrid import rrf


def test_rrf_orders_by_fused_score() -> None:
    # b appears near the top of both lists -> highest fused score
    fused = rrf([["a", "b", "c"], ["b", "c", "d"]], k=60)
    assert fused == ["b", "c", "a", "d"]


def test_rrf_single_list_preserves_order() -> None:
    assert rrf([["x", "y", "z"]]) == ["x", "y", "z"]


def test_rrf_empty_lists() -> None:
    assert rrf([]) == []
    assert rrf([[], []]) == []

"""Retrieval and answer faithfulness evaluation over golden.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paperpilot.config import settings
from paperpilot.llm import complete
from paperpilot.prompts import NOT_FOUND, SYSTEM, format_context
from paperpilot.service import MIN_SCORE
from paperpilot.store import PaperStore

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"
RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"

JUDGE_SYSTEM = (
    "You are a strict fact-checker. Given CONTEXT and ANSWER, decide whether every "
    "factual claim in the ANSWER is supported by the CONTEXT. "
    'Reply with JSON only: {"supported": bool, "unsupported_claims": []}. '
    "Do not include markdown fences or any other text."
)


def load_golden(limit: int | None = None) -> list[dict[str, Any]]:
    """Load golden.jsonl rows (optionally truncated)."""
    rows: list[dict[str, Any]] = []
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def retrieve(
    store: PaperStore,
    question: str,
    paper_id: str,
    k: int,
    retriever: str,
) -> list[dict[str, Any]]:
    """Run the selected retriever; rerank reserved for a later prompt."""
    paper_ids = [paper_id]
    if retriever == "vector":
        return store.search(question, k=k, paper_ids=paper_ids)
    if retriever == "hybrid":
        from paperpilot.hybrid import hybrid_search

        return hybrid_search(
            store, question, paper_ids=paper_ids, k=k, min_score=MIN_SCORE
        )
    if retriever == "rerank":
        from paperpilot.hybrid import hybrid_search
        from paperpilot.rerank import rerank

        candidates = hybrid_search(
            store, question, paper_ids=paper_ids, k=20, min_score=MIN_SCORE
        )
        return rerank(question, candidates, top_n=5)
    raise ValueError(f"Unknown retriever: {retriever}")


def evidence_rank(hits: list[dict[str, Any]], evidence: list[str]) -> int | None:
    """1-based rank of the first hit containing any evidence phrase, else None."""
    if not evidence:
        return None
    phrases = [p.lower() for p in evidence if p]
    for i, hit in enumerate(hits, start=1):
        text = (hit.get("text") or "").lower()
        if any(phrase in text for phrase in phrases):
            return i
    return None


def retrieval_eval(
    rows: list[dict[str, Any]],
    store: PaperStore,
    k: int,
    retriever: str,
) -> dict[str, Any]:
    """Compute hit@k and MRR over answerable golden rows."""
    answerable = [r for r in rows if r.get("answerable")]
    ranks: list[int | None] = []
    misses: list[str] = []

    for row in answerable:
        hits = retrieve(store, row["question"], row["paper_id"], k, retriever)
        rank = evidence_rank(hits, row.get("evidence") or [])
        ranks.append(rank)
        if rank is None:
            misses.append(row["question"])

    n = len(answerable)
    hit_at_k = (sum(1 for r in ranks if r is not None) / n) if n else 0.0
    mrr = (sum((1.0 / r) if r else 0.0 for r in ranks) / n) if n else 0.0
    return {
        "n": n,
        "hit_at_k": hit_at_k,
        "mrr": mrr,
        "misses": misses,
    }


def parse_judge(text: str) -> dict[str, Any] | None:
    """Parse judge JSON; strip fences and a leading ``json`` label."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].lstrip(" \t\r\n:=")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "supported" not in data:
        return None
    return data


def generate_answer(
    store: PaperStore,
    question: str,
    paper_id: str,
    k: int,
    retriever: str,
) -> tuple[str, list[dict[str, Any]], float]:
    """Retrieve, generate with SYSTEM + format_context; return latency."""
    t0 = time.perf_counter()
    hits = retrieve(store, question, paper_id, k, retriever)
    # hybrid/rerank already applied min_score before fusion (BM25-only hits may be 0.0)
    if retriever not in ("hybrid", "rerank"):
        hits = [h for h in hits if h["score"] >= MIN_SCORE]
    if not hits:
        answer = NOT_FOUND
    else:
        context = format_context(hits)
        user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
        answer = complete(SYSTEM, user)
    latency = time.perf_counter() - t0
    return answer, hits, latency


def judge_answer(context: str, answer: str) -> dict[str, Any] | None:
    """Ask the LLM judge; return parsed verdict or None if unparseable."""
    user = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    raw = complete(JUDGE_SYSTEM, user, max_tokens=400)
    return parse_judge(raw)


def answer_eval(
    rows: list[dict[str, Any]],
    store: PaperStore,
    k: int,
    retriever: str,
) -> dict[str, Any]:
    """Faithfulness for answerable rows; refusal accuracy for unanswerable rows."""
    answerable = [r for r in rows if r.get("answerable")]
    unanswerable = [r for r in rows if not r.get("answerable")]
    latencies: list[float] = []
    supported_flags: list[bool] = []
    correct_refusals = 0

    for row in answerable:
        answer, hits, latency = generate_answer(
            store, row["question"], row["paper_id"], k, retriever
        )
        latencies.append(latency)
        context = format_context(hits) if hits else ""
        verdict = judge_answer(context, answer)
        if verdict is None:
            supported_flags.append(False)
        else:
            supported_flags.append(bool(verdict.get("supported")))

    for row in unanswerable:
        answer, _hits, latency = generate_answer(
            store, row["question"], row["paper_id"], k, retriever
        )
        latencies.append(latency)
        if NOT_FOUND in answer:
            correct_refusals += 1

    n_ans = len(answerable)
    n_un = len(unanswerable)
    faithful = (sum(supported_flags) / n_ans) if n_ans else 0.0
    mean_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
    return {
        "faithful": faithful,
        "correct_refusals": correct_refusals,
        "latency_s": mean_latency,
        "n_answerable": n_ans,
        "n_unanswerable": n_un,
    }


def append_results(row: dict[str, Any]) -> None:
    """Append one results row to eval/results.csv (create with header if needed)."""
    fieldnames = [
        "timestamp",
        "chunk_words",
        "chunk_overlap",
        "embed_model",
        "k",
        "retriever",
        "n",
        "hit_at_k",
        "mrr",
        "faithful",
        "correct_refusals",
        "latency_s",
    ]
    write_header = not RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run PaperPilot retrieval/answer eval.")
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument("--limit", type=int, default=None, help="Max golden rows to use")
    parser.add_argument(
        "--skip-answers",
        action="store_true",
        help="Only run retrieval metrics",
    )
    parser.add_argument(
        "--retriever",
        choices=("vector", "hybrid", "rerank"),
        default="vector",
    )
    args = parser.parse_args(argv)

    rows = load_golden(args.limit)
    if not rows:
        print(f"No rows in {GOLDEN_PATH}")
        return

    store = PaperStore()
    ret = retrieval_eval(rows, store, args.k, args.retriever)
    print(f"retrieval n={ret['n']} hit@k={ret['hit_at_k']:.3f} mrr={ret['mrr']:.3f}")
    if ret["misses"]:
        print("misses:")
        for q in ret["misses"]:
            print(f"  - {q}")

    if args.skip_answers:
        ans = {
            "faithful": "",
            "correct_refusals": "",
            "latency_s": "",
        }
    else:
        ans = answer_eval(rows, store, args.k, args.retriever)
        print(
            f"answer faithful={ans['faithful']:.3f} "
            f"correct_refusals={ans['correct_refusals']}/"
            f"{ans['n_unanswerable']} "
            f"latency_s={ans['latency_s']:.2f}"
        )

    append_results(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chunk_words": settings.chunk_words,
            "chunk_overlap": settings.chunk_overlap,
            "embed_model": settings.embed_model,
            "k": args.k,
            "retriever": args.retriever,
            "n": ret["n"],
            "hit_at_k": f"{ret['hit_at_k']:.4f}",
            "mrr": f"{ret['mrr']:.4f}",
            "faithful": "" if ans["faithful"] == "" else f"{ans['faithful']:.4f}",
            "correct_refusals": ans["correct_refusals"],
            "latency_s": "" if ans["latency_s"] == "" else f"{ans['latency_s']:.3f}",
        }
    )
    print(f"appended {RESULTS_PATH}")


if __name__ == "__main__":
    main()

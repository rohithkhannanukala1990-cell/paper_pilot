"""Print random judge verdicts beside answer and context for manual review."""

from __future__ import annotations

import argparse
import random

from paperpilot.config import settings
from paperpilot.prompts import format_context
from paperpilot.store import PaperStore

from eval.run_eval import generate_answer, judge_answer, load_golden

DEFAULT_N = 10


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Spot-check LLM judge verdicts.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of samples")
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rows = [r for r in load_golden() if r.get("answerable")]
    if not rows:
        print("No answerable rows in golden.jsonl")
        return

    rng = random.Random(args.seed)
    sample = rows if len(rows) <= args.n else rng.sample(rows, args.n)
    store = PaperStore()

    for i, row in enumerate(sample, start=1):
        answer, hits, _latency = generate_answer(
            store, row["question"], row["paper_id"], args.k, "vector"
        )
        context = format_context(hits) if hits else "(no hits above MIN_SCORE)"
        verdict = judge_answer(context if hits else "", answer)
        print("=" * 72)
        print(f"[{i}] Q: {row['question']}")
        print(f"paper_id={row['paper_id']}")
        print("- answer -")
        print(answer)
        print("- context -")
        print(context)
        print("- judge -")
        print(verdict if verdict is not None else "UNPARSEABLE")
        print()


if __name__ == "__main__":
    main()

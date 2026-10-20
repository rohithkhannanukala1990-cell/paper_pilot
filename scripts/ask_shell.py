"""CLI: ask a question via PaperPilot and print answer, sources, citations."""

from __future__ import annotations

import argparse

from paperpilot.service import PaperPilot


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ask a question over indexed papers.")
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument(
        "--paper-id",
        action="append",
        dest="paper_ids",
        default=None,
        help="Restrict retrieval to this paper_id (repeatable)",
    )
    args = parser.parse_args(argv)

    result = PaperPilot().ask(args.question, paper_ids=args.paper_ids)

    print(result["answer"])
    if result["sources"]:
        print("\nSources:")
        for i, src in enumerate(result["sources"], start=1):
            meta = src.get("meta") or {}
            print(
                f"  [{i}] {meta.get('title', '')} | {meta.get('section', '')} | "
                f"p.{meta.get('page', '')} | score={src['score']:.3f}"
            )
    if result["invalid_citations"]:
        print(f"\nInvalid citations: {result['invalid_citations']}")


if __name__ == "__main__":
    main()

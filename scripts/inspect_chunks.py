"""Print chunk statistics for a PDF path given as argv[1]."""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

from paperpilot.ingest import ingest_pdf


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <pdf_path>", file=sys.stderr)
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    chunks = ingest_pdf(pdf_path)
    if not chunks:
        print("No chunks produced.")
        return

    word_counts = [len(c.text.split()) for c in chunks]
    avg_words = sum(word_counts) / len(word_counts)
    sections = Counter(c.section for c in chunks)

    print(f"chunks: {len(chunks)}")
    print(f"avg words/chunk: {avg_words:.1f}")
    print("section distribution:")
    for name, count in sections.most_common():
        print(f"  {name}: {count}")

    sample = random.sample(chunks, k=min(3, len(chunks)))
    print("\n3 random chunks:")
    for c in sample:
        preview = c.text if len(c.text) <= 240 else c.text[:237] + "..."
        print(f"\n[{c.chunk_id}] section={c.section!r} page={c.page} words={len(c.text.split())}")
        print(preview)


if __name__ == "__main__":
    main()

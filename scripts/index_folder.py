"""Ingest every PDF under data/papers, skipping papers already in the store."""

from __future__ import annotations

import sys
from pathlib import Path

from paperpilot.config import settings
from paperpilot.ingest import file_id, ingest_pdf
from paperpilot.store import PaperStore


def main() -> None:
    folder = Path(settings.upload_dir)
    if not folder.is_dir():
        print(f"Upload dir not found: {folder}", file=sys.stderr)
        sys.exit(1)

    store = PaperStore()
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {folder}")
        return

    for pdf in pdfs:
        paper_id = file_id(pdf)
        if store.has_paper(paper_id):
            print(f"skip {pdf.name} ({paper_id})")
            continue
        chunks = ingest_pdf(pdf)
        store.add_chunks(chunks)
        print(f"indexed {pdf.name} ({paper_id}): {len(chunks)} chunks")


if __name__ == "__main__":
    main()

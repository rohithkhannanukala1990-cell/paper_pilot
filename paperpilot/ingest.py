"""PDF parsing: extract cleaned paragraph blocks with section labels.

Known limitations:
- Section boundaries rely on a regex heading list; nonstandard headings are missed.
- Tables and equations are treated as plain text and may be fragmented or noisy.
- Scanned / image-only PDFs yield little or no text without OCR.
- ``clean`` joins lines ending in ``-\\n``, which can wrongly merge real hyphens
  at line breaks (e.g. compound words split across lines).
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz

from paperpilot.config import settings

logger = logging.getLogger(__name__)

HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(abstract|introduction|related work|background|method(?:s)?|"
    r"methodology|approach|experiment(?:s)?|results|discussion|"
    r"limitations|conclusion(?:s)?|references)\s*$",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    """A text chunk tagged with paper and section metadata."""

    chunk_id: str
    paper_id: str
    title: str
    section: str
    page: int
    text: str


def file_id(path: str | Path) -> str:
    """Return the first 12 hex chars of the SHA-1 of the file bytes."""
    data = Path(path).read_bytes()
    return hashlib.sha1(data).hexdigest()[:12]


def clean(text: str) -> str:
    """Remove hyphenated line breaks, turn newlines into spaces, collapse whitespace."""
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def guess_title(doc: fitz.Document, path: str | Path) -> str:
    """Prefer PDF metadata title, else a page-1 text block, else the file stem."""
    meta_title = (doc.metadata or {}).get("title") or ""
    meta_title = meta_title.strip()
    if len(meta_title) > 8:
        return meta_title

    if doc.page_count > 0:
        page = doc.load_page(0)
        for blk in page.get_text("blocks", sort=True):
            if blk[6] != 0:
                continue
            candidate = clean(blk[4])
            if 15 <= len(candidate) <= 200:
                return candidate

    return Path(path).stem


def parse_pdf(path: str | Path) -> tuple[str, str, list[tuple[str, str, int]]]:
    """Parse a PDF into ``(paper_id, title, [(paragraph, section, page), ...])``."""
    path = Path(path)
    paper_id = file_id(path)
    paragraphs: list[tuple[str, str, int]] = []

    with fitz.open(path) as doc:
        title = guess_title(doc, path)
        section = "Front Matter"

        for page in doc:
            page_number = page.number + 1
            for blk in page.get_text("blocks", sort=True):
                if blk[6] != 0:
                    continue
                text = clean(blk[4])
                if not text:
                    continue

                heading = HEADING.match(text)
                if heading:
                    name = heading.group(1)
                    if name.lower() == "references":
                        logger.debug("Stopping at References on page %s", page_number)
                        return paper_id, title, paragraphs
                    section = " ".join(w.capitalize() for w in name.split())
                    continue

                if len(text) < 25:
                    continue
                paragraphs.append((text, section, page_number))

    return paper_id, title, paragraphs


def chunk(
    paragraphs: list[tuple[str, str, int]],
    pid: str,
    title: str,
    target: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Pack paragraphs into Chunks; never cross sections; overlap only within a section."""
    if target is None:
        target = settings.chunk_words
    if overlap is None:
        overlap = settings.chunk_overlap

    out: list[Chunk] = []
    index = 0

    def emit(
        words: list[str],
        section: str,
        page: int,
        overlap_prefix: int,
    ) -> list[str]:
        """Emit one chunk if valid; return overlap carry (empty if dropped or none)."""
        nonlocal index
        if not words or len(words) <= overlap_prefix:
            return []
        if len(words) < 30:
            return []
        out.append(
            Chunk(
                chunk_id=f"{pid}-{index}",
                paper_id=pid,
                title=title,
                section=section,
                page=page,
                text=" ".join(words),
            )
        )
        index += 1
        if overlap <= 0:
            return []
        return words[-overlap:]

    i = 0
    while i < len(paragraphs):
        section = paragraphs[i][1]
        group: list[tuple[str, str, int]] = []
        while i < len(paragraphs) and paragraphs[i][1] == section:
            group.append(paragraphs[i])
            i += 1

        buf: list[str] = []
        start_page: int | None = None
        last_page = group[0][2]
        overlap_prefix = 0

        for text, _sec, page in group:
            words = text.split()
            if not words:
                continue
            last_page = page

            if len(words) > target:
                if buf:
                    carry = emit(buf, section, start_page or page, overlap_prefix)
                    buf = list(carry)
                    overlap_prefix = len(buf)
                    start_page = last_page if buf else None

                pos = 0
                while pos < len(words):
                    room = target - len(buf)
                    if room <= 0:
                        carry = emit(buf, section, start_page or page, overlap_prefix)
                        buf = list(carry)
                        overlap_prefix = len(buf)
                        start_page = page if buf else None
                        room = target - len(buf)
                        if room <= 0:
                            room = target

                    take = words[pos : pos + room]
                    pos += len(take)
                    if not buf:
                        start_page = page
                    buf.extend(take)

                    if len(buf) >= target:
                        carry = emit(buf, section, start_page or page, overlap_prefix)
                        buf = list(carry)
                        overlap_prefix = len(buf)
                        start_page = page if buf else None
                continue

            if buf and len(buf) + len(words) > target:
                carry = emit(buf, section, start_page or page, overlap_prefix)
                buf = list(carry)
                overlap_prefix = len(buf)
                start_page = last_page if buf else None

            if not buf:
                start_page = page
            buf.extend(words)

        if buf:
            emit(buf, section, start_page or last_page, overlap_prefix)

    return out


def ingest_pdf(path: str | Path) -> list[Chunk]:
    """Parse a PDF and return packed Chunks."""
    paper_id, title, paragraphs = parse_pdf(path)
    return chunk(paragraphs, paper_id, title)

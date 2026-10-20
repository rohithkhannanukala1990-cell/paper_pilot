"""Offline tests for text cleaning and chunk packing."""

from __future__ import annotations

from paperpilot.ingest import clean, chunk


def test_clean_joins_hyphenated_line_breaks() -> None:
    assert clean("well-\nknown idea") == "wellknown idea"
    assert clean("multi-\nline\n\ntext") == "multiline text"


def test_chunk_never_crosses_sections() -> None:
    paras = [
        (" ".join(f"i{n}" for n in range(80)), "Introduction", 1),
        (" ".join(f"m{n}" for n in range(80)), "Methods", 2),
    ]
    chunks = chunk(paras, "pid", "Title", target=100, overlap=20)
    assert chunks
    for c in chunks:
        words = c.text.split()
        assert all(w.startswith("i") for w in words) or all(
            w.startswith("m") for w in words
        )


def test_chunk_keeps_size_page_and_drops_short_leftovers() -> None:
    # 50 + 50 packs to target; with overlap=0 the 20-word remnant is dropped (<30)
    paras = [
        (" ".join(f"a{n}" for n in range(50)), "Abstract", 1),
        (" ".join(f"b{n}" for n in range(50)), "Abstract", 1),
        (" ".join(f"c{n}" for n in range(20)), "Abstract", 2),
    ]
    chunks = chunk(paras, "pid", "Title", target=100, overlap=0)
    assert len(chunks) == 1
    assert len(chunks[0].text.split()) <= 100
    assert chunks[0].page == 1
    assert all(len(c.text.split()) >= 30 for c in chunks)


def test_chunk_overlap_does_not_cross_sections() -> None:
    """300 Intro words then 100 Methods -> Methods chunk is exactly 100 words."""
    intro = (" ".join(f"i{n}" for n in range(300)), "Introduction", 1)
    methods = (" ".join(f"m{n}" for n in range(100)), "Methods", 5)
    chunks = chunk([intro, methods], "pid", "Title", target=220, overlap=40)

    methods_chunks = [c for c in chunks if c.section == "Methods"]
    assert len(methods_chunks) == 1
    assert len(methods_chunks[0].text.split()) == 100
    assert all(w.startswith("m") for w in methods_chunks[0].text.split())
    assert methods_chunks[0].page == 5

    intro_chunks = [c for c in chunks if c.section == "Introduction"]
    assert intro_chunks
    assert all(len(c.text.split()) <= 220 for c in intro_chunks)
    assert all(w.startswith("i") for c in intro_chunks for w in c.text.split())

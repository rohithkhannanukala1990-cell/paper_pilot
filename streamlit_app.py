"""Streamlit UI for PaperPilot (thin layer — calls PaperPilot only)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from paperpilot.config import settings
from paperpilot.service import PaperPilot

st.set_page_config(page_title="PaperPilot", layout="wide")


@st.cache_resource
def get_pilot() -> PaperPilot:
    """Cached PaperPilot instance for the Streamlit process."""
    return PaperPilot()


def show_sources(hits: list[dict], invalid: list[int] | None = None) -> None:
    """Warn on invalid citations, then expand each source."""
    if invalid:
        st.warning(f"Invalid citations: {invalid}")
    for i, hit in enumerate(hits, start=1):
        meta = hit.get("meta") or {}
        title = hit.get("title") or meta.get("title", "")
        section = hit.get("section") or meta.get("section", "")
        page = hit.get("page", meta.get("page", ""))
        score = float(hit.get("score", 0.0))
        label = f"[{i}] {title} | {section} | p.{page} | score {score:.2f}"
        with st.expander(label):
            text = hit.get("text")
            if text:
                st.write(text)


def _init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "indexed" not in st.session_state:
        st.session_state.indexed = set()


def _selected_paper_ids(papers: list[dict]) -> list[str] | None:
    selected: list[str] = []
    for paper in papers:
        pid = paper["paper_id"]
        label = f"{paper['title']} ({pid}, {paper['chunks']} chunks)"
        if st.checkbox(label, key=f"scope_{pid}", value=False):
            selected.append(pid)
    return selected or None


def main() -> None:
    _init_state()
    pilot = get_pilot()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    with st.sidebar:
        st.header("Library")
        uploads = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploads:
            for uploaded in uploads:
                if uploaded.name in st.session_state.indexed:
                    continue
                dest = upload_dir / uploaded.name
                dest.write_bytes(uploaded.getvalue())
                try:
                    info = pilot.add_pdf(dest)
                    st.session_state.indexed.add(uploaded.name)
                    st.session_state.indexed.add(info["paper_id"])
                    st.success(
                        f"Indexed {uploaded.name}: {info['chunks']} chunks "
                        f"({info['paper_id']})"
                    )
                except ValueError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(str(exc))

        st.subheader("Search scope")
        papers = pilot.store.list_papers()
        if not papers:
            st.caption("No papers indexed yet.")
            selected = None
        else:
            selected = _selected_paper_ids(papers)
            st.caption("Leave all unchecked to search the whole library.")

        st.subheader("Delete")
        for paper in papers:
            pid = paper["paper_id"]
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.write(f"{paper['title']} (`{pid}`)")
            with col_b:
                if st.button("Delete", key=f"del_{pid}"):
                    pilot.store.delete_paper(pid)
                    st.session_state.indexed.discard(pid)
                    st.rerun()

    st.title("PaperPilot")
    tab_ask, tab_sum, tab_cmp = st.tabs(["Ask", "Summarize", "Compare"])

    with tab_ask:
        for role, text in st.session_state.history:
            with st.chat_message(role):
                st.markdown(text)

        if prompt := st.chat_input("Ask a question about the papers"):
            prior = list(st.session_state.history)
            st.session_state.history.append(("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    result = pilot.ask(
                        prompt,
                        paper_ids=selected,
                        history=prior,
                    )
                    st.markdown(result["answer"])
                    show_sources(result["sources"], result["invalid_citations"])
                    st.session_state.history.append(("assistant", result["answer"]))
                except Exception as exc:
                    st.error(str(exc))
                    st.session_state.history.pop()

    with tab_sum:
        if not papers:
            st.info("Index a paper first.")
        else:
            options = {f"{p['title']} ({p['paper_id']})": p["paper_id"] for p in papers}
            choice = st.selectbox("Paper", list(options.keys()))
            if st.button("Summarize", type="primary"):
                paper_id = options[choice]
                try:
                    result = pilot.summarize(paper_id)
                    st.subheader("TL;DR")
                    st.write(result["tldr"])
                    for section in result["sections"]:
                        st.subheader(section["name"])
                        st.write(section["text"])
                        show_sources(section["hits"], None)
                except Exception as exc:
                    st.error(str(exc))

    with tab_cmp:
        aspects_raw = st.text_input(
            "Aspects (comma-separated)",
            value="problem, method, datasets, results, limitations",
        )
        aspects = [a.strip() for a in aspects_raw.split(",") if a.strip()]
        scope = selected or []
        if len(scope) < 2:
            st.info("Select at least two papers in the sidebar to compare.")
        elif st.button("Compare", type="primary"):
            try:
                result = pilot.compare(scope, aspects)
                st.markdown(result["answer"])
                show_sources(result["sources"], result["invalid_citations"])
            except Exception as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()

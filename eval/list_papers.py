"""Print paper_id and title for every indexed paper (for filling golden.jsonl)."""

from __future__ import annotations

from paperpilot.store import PaperStore


def main() -> None:
    papers = PaperStore().list_papers()
    if not papers:
        print("No papers indexed. Run scripts/index_folder.py first.")
        return
    for paper in papers:
        print(f"{paper['paper_id']}\t{paper['title']}")


if __name__ == "__main__":
    main()

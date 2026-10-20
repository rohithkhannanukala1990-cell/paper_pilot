"""FastAPI surface for PaperPilot (thin layer over the service)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from paperpilot.config import MAX_UPLOAD_MB, settings
from paperpilot.llm import LLMError
from paperpilot.service import PaperPilot

app = FastAPI(title="PaperPilot API", version="1.0")

_CHUNK_SIZE = 1024 * 1024


@lru_cache
def get_pilot() -> PaperPilot:
    """Lazy singleton PaperPilot; override in tests via ``app.dependency_overrides``."""
    return PaperPilot()


@app.exception_handler(LLMError)
async def llm_error_handler(_request, exc: LLMError) -> JSONResponse:
    """Map provider failures that survive retries to HTTP 503."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


class AskIn(BaseModel):
    question: str = Field(min_length=3)
    paper_ids: list[str] | None = None
    history: list[tuple[str, str]] = Field(default_factory=list)


class CompareIn(BaseModel):
    paper_ids: list[str] = Field(min_length=2, max_length=5)
    aspects: list[str] = Field(
        default_factory=lambda: [
            "problem",
            "method",
            "datasets",
            "results",
            "limitations",
        ]
    )


@app.get("/health")
def health() -> dict[str, bool]:
    """Liveness check."""
    return {"ok": True}


@app.get("/papers")
def list_papers(pilot: PaperPilot = Depends(get_pilot)) -> list[dict]:
    """List indexed papers."""
    return pilot.store.list_papers()


@app.delete("/papers/{paper_id}")
def delete_paper(paper_id: str, pilot: PaperPilot = Depends(get_pilot)) -> dict[str, str]:
    """Delete all chunks for ``paper_id``."""
    pilot.store.delete_paper(paper_id)
    return {"deleted": paper_id}


@app.post("/papers")
async def upload_paper(
    file: UploadFile = File(...),
    pilot: PaperPilot = Depends(get_pilot),
) -> dict:
    """Upload and index a PDF."""
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Filename must end in .pdf")

    safe_name = os.path.basename(filename)
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / safe_name
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    written = 0

    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(_CHUNK_SIZE)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_MB} MB upload limit",
                    )
                out.write(chunk)
        return pilot.add_pdf(dest)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/ask")
def ask(body: AskIn, pilot: PaperPilot = Depends(get_pilot)) -> dict:
    """Answer a question with retrieved citations."""
    return pilot.ask(
        body.question,
        paper_ids=body.paper_ids or None,
        history=body.history or None,
    )


@app.get("/papers/{paper_id}/summary")
def summarize_paper(paper_id: str, pilot: PaperPilot = Depends(get_pilot)) -> dict:
    """Summarize one indexed paper by facet."""
    known = {p["paper_id"] for p in pilot.store.list_papers()}
    if paper_id not in known:
        raise HTTPException(status_code=404, detail=f"Unknown paper_id: {paper_id}")
    return pilot.summarize(paper_id)


@app.post("/compare")
def compare(body: CompareIn, pilot: PaperPilot = Depends(get_pilot)) -> dict:
    """Compare 2–5 papers on the given aspects."""
    return pilot.compare(body.paper_ids, body.aspects)

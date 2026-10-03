"""HTTP API. Current endpoints: /health and /search (BM25). /ask (grounded generation) comes later.

The index is loaded eagerly at startup: if the corpus is missing the process crashes, so a bad
image never starts serving traffic (Cloud Run keeps the previous revision live).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from ukmoney_rag import __version__
from ukmoney_rag.search import BM25Index

DISCLAIMER = (
    "Results are excerpts of GOV.UK guidance (Open Government Licence v3.0). This service is not "
    "affiliated with or endorsed by the UK government, and is not financial advice."
)


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    git_sha: str
    n_chunks: int
    corpus_sha256: str | None
    corpus_created_at: str | None


class SearchResult(BaseModel):
    chunk_id: str
    score: float
    title: str
    section_title: str
    url: str
    text: str
    licence: str
    updated_at: datetime | None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    disclaimer: str


def create_app(data_dir: Path | None = None) -> FastAPI:
    data_dir = data_dir or Path(os.environ.get("UKMONEY_DATA_DIR", "data/processed"))
    chunks_path = data_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"{chunks_path} not found; run `make ingest` first")
    index = BM25Index.from_jsonl(chunks_path)
    manifest_path = data_dir / "manifest.json"
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )

    app = FastAPI(
        title="UK Money Guide",
        version=__version__,
        description="Search over GOV.UK money guidance. " + DISCLAIMER,
    )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health")
    def health() -> Health:
        return Health(
            status="ok",
            version=__version__,
            git_sha=os.environ.get("GIT_SHA", "unknown"),
            n_chunks=len(index),
            corpus_sha256=manifest.get("chunks_sha256"),
            corpus_created_at=manifest.get("created_at"),
        )

    @app.get("/search")
    def search(
        q: Annotated[str, Query(min_length=2, max_length=300, description="Question or keywords")],
        k: Annotated[int, Query(ge=1, le=20)] = 5,
    ) -> SearchResponse:
        hits = index.search(q, k)
        return SearchResponse(
            query=q,
            results=[
                SearchResult(
                    chunk_id=h.chunk.chunk_id,
                    score=h.score,
                    title=h.chunk.title,
                    section_title=h.chunk.section_title,
                    url=h.chunk.url,
                    text=h.chunk.text,
                    licence=h.chunk.licence,
                    updated_at=h.chunk.updated_at,
                )
                for h in hits
            ],
            disclaimer=DISCLAIMER,
        )

    return app


def main() -> None:
    """Container entrypoint. Cloud Run injects PORT; default 8080 for local runs."""
    import uvicorn

    uvicorn.run(
        create_app(),
        host="0.0.0.0",  # required inside a container
        port=int(os.environ.get("PORT", "8080")),
        proxy_headers=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()

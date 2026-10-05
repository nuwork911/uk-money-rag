"""HTTP API: /health and /search (BM25 or dense). /ask (grounded generation) comes later.

Everything is loaded and cross-checked eagerly at startup. A missing corpus, a manifest that
disagrees with the corpus bytes, or a dense index built for a different corpus all crash the
process, so a bad image never starts serving traffic (Cloud Run keeps the previous revision).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from ukmoney_rag import __version__
from ukmoney_rag.api_dense import DenseRetriever, SearchMode
from ukmoney_rag.dense import IndexLoadError, load_chunks
from ukmoney_rag.models import Chunk
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
    corpus_sha256: str = Field(description="sha256 of the served chunks.jsonl bytes")
    corpus_created_at: str | None
    retrievers: list[SearchMode]
    dense_index_id: str | None
    embedding_model: str | None


class SearchResult(BaseModel):
    chunk_id: str
    score: float = Field(description="Retriever-specific; not comparable across modes")
    title: str
    section_title: str
    url: str
    text: str
    licence: str
    updated_at: datetime | None = Field(
        description=(
            "Last *major* update reported by GOV.UK. Minor edits (including some rate "
            "changes) do not move it, so this is not a freshness guarantee."
        )
    )


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    results: list[SearchResult]
    disclaimer: str


def _result(chunk: Chunk, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk.chunk_id,
        score=score,
        title=chunk.title,
        section_title=chunk.section_title,
        url=chunk.url,
        text=chunk.text,
        licence=chunk.licence,
        updated_at=chunk.updated_at,
    )


def create_app(data_dir: Path | None = None, dense: DenseRetriever | None = None) -> FastAPI:
    """Build the app. Dense retrieval is on if `dense` is given or UKMONEY_DENSE=1."""
    data_dir = data_dir or Path(os.environ.get("UKMONEY_DATA_DIR", "data/snapshot"))
    chunks_path = data_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"{chunks_path} not found; run `make ingest && make snapshot`")
    chunks, corpus_sha256 = load_chunks(chunks_path)  # hash of the bytes actually served

    manifest_path = data_dir / "manifest.json"
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    claimed = manifest.get("chunks_sha256")
    if claimed is not None and claimed != corpus_sha256:
        raise IndexLoadError(
            f"{manifest_path} says chunks_sha256={claimed[:12]}, file bytes are "
            f"{corpus_sha256[:12]}; corpus and manifest are out of sync"
        )

    if dense is None and os.environ.get("UKMONEY_DENSE") == "1":
        dense = DenseRetriever.from_env(corpus_sha256)  # raises if index is missing/mismatched
    if dense is not None and dense.index.manifest.corpus_sha256 != corpus_sha256:
        raise IndexLoadError(
            f"dense index {dense.index.manifest.index_id} is for a different corpus than "
            f"{chunks_path}"
        )

    bm25 = BM25Index(chunks)
    by_id = {c.chunk_id: c for c in chunks}
    retrievers: list[SearchMode] = ["bm25", "dense"] if dense else ["bm25"]

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
            n_chunks=len(chunks),
            corpus_sha256=corpus_sha256,
            corpus_created_at=manifest.get("created_at"),
            retrievers=retrievers,
            dense_index_id=dense.index.manifest.index_id if dense else None,
            embedding_model=dense.index.manifest.model_id if dense else None,
        )

    @app.get("/search")
    def search(
        q: Annotated[str, Query(min_length=2, max_length=300, description="Question or keywords")],
        k: Annotated[int, Query(ge=1, le=20)] = 5,
        mode: Annotated[SearchMode, Query(description="Retriever to use")] = "bm25",
    ) -> SearchResponse:
        if mode == "dense":
            if dense is None:
                raise HTTPException(503, "dense retrieval is not enabled on this deployment")
            results = [_result(by_id[cid], round(s, 4)) for cid, s in dense.search(q, k)]
        else:
            results = [_result(h.chunk, h.score) for h in bm25.search(q, k)]
        return SearchResponse(query=q, mode=mode, results=results, disclaimer=DISCLAIMER)

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

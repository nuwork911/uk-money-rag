"""Dense retrieval: embedder + index validated together at startup. Used by api.py."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from ukmoney_rag.dense import DenseIndex, IndexLoadError, resolve_index_dir
from ukmoney_rag.embed import Embedder, FastEmbedEmbedder

SearchMode = Literal["bm25", "dense"]


class DenseRetriever:
    """Owns the embedder + index; validates they belong together at startup, not per request."""

    def __init__(self, index: DenseIndex, embedder: Embedder) -> None:
        m = index.manifest
        if (m.model_id, m.model_sha256, m.query_prefix) != (
            embedder.model_id,
            embedder.model_sha256,
            embedder.query_prefix,
        ):
            raise IndexLoadError(
                f"index {m.index_id} was built with {m.model_id}@{m.model_sha256[:12]}, "
                f"serving embedder is {embedder.model_id}@{embedder.model_sha256[:12]}"
            )
        self.index = index
        self.embedder = embedder

    @classmethod
    def from_env(cls, corpus_sha256: str, embedder: Embedder | None = None) -> DenseRetriever:
        root = Path(os.environ.get("UKMONEY_INDEX_ROOT", "indexes"))
        # Cheap checks first: a missing/mismatched index fails in ms, before the model loads.
        index = DenseIndex.load(resolve_index_dir(root), corpus_sha256)
        if embedder is None:
            cache = Path(os.environ.get("UKMONEY_MODEL_CACHE", ".cache/models"))
            embedder = FastEmbedEmbedder(cache_dir=cache)
        return cls(index, embedder)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.index.search(self.embedder.embed_query(query), k)

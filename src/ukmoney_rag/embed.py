"""Text embedding behind a small Protocol, so tests and the index never depend on a real model.

Model choice (see docs/adr/0002-embedding-model.md): BAAI/bge-small-en-v1.5 via fastembed's
quantised ONNX export. 384-d, ~65 MB on disk, no PyTorch, so the Cloud Run image stays small.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float32]

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# BGE v1.5 is trained with this instruction on the *query* side only. fastembed does NOT add it
# (its query_embed() is plain embed() for this model), so we add it ourselves and record it in
# the index manifest: changing it silently would change every query vector.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Pinned fingerprint of model_optimized.onnx (Qdrant/bge-small-en-v1.5-onnx-Q). If the upstream
# repo is ever re-uploaded, index builds fail loudly instead of drifting.
DEFAULT_MODEL_SHA256 = "51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431"


class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def model_sha256(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def query_prefix(self) -> str: ...

    def embed_passages(self, texts: Sequence[str]) -> Matrix: ...

    def embed_query(self, text: str) -> Matrix: ...


def l2_normalize(m: Matrix) -> Matrix:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalise a zero vector")
    return (m / norms).astype(np.float32)


def find_onnx_file(cache_dir: Path) -> Path:
    files = sorted(cache_dir.rglob("*.onnx"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected exactly one .onnx file under {cache_dir}, got {files}")
    return files[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class FastEmbedEmbedder:
    """fastembed-backed embedder. Verifies the model file hash on load."""

    def __init__(
        self,
        cache_dir: Path,
        model_id: str = DEFAULT_MODEL,
        expected_sha256: str | None = DEFAULT_MODEL_SHA256,
        query_prefix: str = BGE_QUERY_PREFIX,
        threads: int | None = None,
    ) -> None:
        from fastembed import TextEmbedding  # heavy import kept off the BM25-only path

        self._model = TextEmbedding(model_id, cache_dir=str(cache_dir), threads=threads)
        self._model_id = model_id
        self._model_sha256 = sha256_file(find_onnx_file(cache_dir))
        if expected_sha256 is not None and self._model_sha256 != expected_sha256:
            raise RuntimeError(
                f"model file hash {self._model_sha256} != pinned {expected_sha256}; "
                "upstream model changed. Re-pin deliberately and rebuild the index."
            )
        self._query_prefix = query_prefix
        self._dim = int(self.embed_passages(["dimension probe"]).shape[1])

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def query_prefix(self) -> str:
        return self._query_prefix

    def embed_passages(self, texts: Sequence[str]) -> Matrix:
        vecs = np.asarray(list(self._model.embed(list(texts), batch_size=32)), dtype=np.float32)
        return l2_normalize(vecs)

    def embed_query(self, text: str) -> Matrix:
        return self.embed_passages([self._query_prefix + text])

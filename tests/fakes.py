"""Test doubles shared across test modules (pytest pythonpath includes tests/)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from ukmoney_rag.embed import Matrix, l2_normalize
from ukmoney_rag.models import Chunk

CORPUS = "a" * 64


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: no model download, real geometry.

    Shared words => higher cosine, which is enough to test ranking logic offline.
    """

    def __init__(self, dim: int = 64, query_prefix: str = "q: ", model_sha256: str = "f" * 64):
        self._dim = dim
        self._prefix = query_prefix
        self._sha = model_sha256

    model_id = "fake/hash-bow"

    @property
    def model_sha256(self) -> str:
        return self._sha

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def query_prefix(self) -> str:
        return self._prefix

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self._dim] += 1.0
        v[0] += 1e-3  # avoid zero vectors for empty text
        return v

    def embed_passages(self, texts: Sequence[str]) -> Matrix:
        return l2_normalize(np.stack([self._one(t) for t in texts]))

    def embed_query(self, text: str) -> Matrix:
        return self.embed_passages([text])  # prefix recorded, not applied: keeps tests readable


def make_chunk(i: int, title: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"doc{i}:{i:012d}",
        doc_id=f"doc{i}",
        chunk_index=0,
        url=f"https://www.gov.uk/doc{i}",
        title=title,
        section_title="Overview",
        text=text,
        n_words=len(text.split()),
        licence="OGL-UK-3.0",
        updated_at=None,
    )

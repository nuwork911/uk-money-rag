"""Versioned dense index: build, persist, load (with validation), search.

On-disk layout (one immutable directory per index version):

    indexes/<index_id>/
        manifest.json     what produced this index (inputs, model, versions)
        embeddings.npy    float32 [n_chunks, dim], L2-normalised, row i <-> chunk_ids[i]
        chunk_ids.json    row order

``index_id`` is derived from the *inputs* (corpus hash, model hash, query prefix, text field,
schema version), never from the output bytes: ONNX float output can differ in the last bits
across CPUs, so an output hash would make the same index look like a different version.

Search is exact brute-force cosine (one matrix-vector product). See docs/adr/0003-vector-store.md
for why that beats a vector DB at this corpus size, and when that stops being true.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from ukmoney_rag.embed import Embedder, Matrix
from ukmoney_rag.models import Chunk

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
TEXT_FIELD: Literal["embed_text"] = "embed_text"


class IndexManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int
    index_id: str
    corpus_sha256: str
    n_chunks: int
    model_id: str
    model_sha256: str
    dim: int
    query_prefix: str
    text_field: Literal["embed_text"]
    created_at: datetime
    fastembed_version: str | None
    embeddings_sha256: str  # informational only; NOT part of index_id (see module docstring)


class IndexLoadError(RuntimeError):
    """Index on disk is missing, corrupt, or doesn't match the corpus being served."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_chunks(path: Path) -> tuple[list[Chunk], str]:
    """Return chunks and the sha256 of the exact file bytes (the corpus identity)."""
    raw = path.read_bytes()
    chunks = [Chunk.model_validate_json(line) for line in raw.decode("utf-8").splitlines() if line]
    if not chunks:
        raise IndexLoadError(f"{path} contains no chunks")
    return chunks, sha256_bytes(raw)


def compute_index_id(
    corpus_sha256: str, model_id: str, model_sha256: str, query_prefix: str
) -> str:
    key = "\x1f".join(
        [str(SCHEMA_VERSION), corpus_sha256, model_id, model_sha256, query_prefix, TEXT_FIELD]
    )
    return sha256_bytes(key.encode())[:12]


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _write_latest(out_root: Path, index_id: str) -> None:
    (out_root / "LATEST").write_text(index_id + "\n", encoding="utf-8")


def build_index(
    chunks: Sequence[Chunk], corpus_sha256: str, embedder: Embedder, out_root: Path
) -> Path:
    """Embed chunks and write a new index version. Idempotent: returns existing dir if present."""
    index_id = compute_index_id(
        corpus_sha256, embedder.model_id, embedder.model_sha256, embedder.query_prefix
    )
    final = out_root / index_id
    if final.exists():
        logger.info("index %s already exists; not rebuilding", index_id)
        _write_latest(out_root, index_id)
        return final

    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        raise IndexLoadError("duplicate chunk_id in corpus")
    emb = embedder.embed_passages([c.embed_text for c in chunks])
    if emb.shape != (len(chunks), embedder.dim):
        raise IndexLoadError(
            f"embedder returned {emb.shape}, expected {(len(chunks), embedder.dim)}"
        )

    manifest = IndexManifest(
        schema_version=SCHEMA_VERSION,
        index_id=index_id,
        corpus_sha256=corpus_sha256,
        n_chunks=len(chunks),
        model_id=embedder.model_id,
        model_sha256=embedder.model_sha256,
        dim=embedder.dim,
        query_prefix=embedder.query_prefix,
        text_field=TEXT_FIELD,
        created_at=datetime.now(UTC),
        fastembed_version=_pkg_version("fastembed"),
        embeddings_sha256=sha256_bytes(emb.tobytes()),
    )

    # Write to a temp dir and rename, so a crash never leaves a half-written "valid" index.
    out_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out_root, prefix=".tmp-"))
    try:
        np.save(tmp / "embeddings.npy", emb, allow_pickle=False)
        (tmp / "chunk_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        (tmp / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    _write_latest(out_root, index_id)
    logger.info("built index %s: %d x %d", index_id, *emb.shape)
    return final


class DenseIndex:
    def __init__(self, manifest: IndexManifest, chunk_ids: list[str], matrix: Matrix) -> None:
        self.manifest = manifest
        self.chunk_ids = chunk_ids
        self.matrix = matrix

    @classmethod
    def load(cls, path: Path, expected_corpus_sha256: str) -> DenseIndex:
        """Load and validate. Refuses to serve an index built from a different corpus."""
        try:
            manifest = IndexManifest.model_validate_json(
                (path / "manifest.json").read_text(encoding="utf-8")
            )
            ids: list[str] = json.loads((path / "chunk_ids.json").read_text(encoding="utf-8"))
            matrix = np.load(path / "embeddings.npy", allow_pickle=False)
        except (OSError, ValueError) as e:
            raise IndexLoadError(f"cannot load index at {path}: {e}") from e
        if manifest.schema_version != SCHEMA_VERSION:
            raise IndexLoadError(f"index schema {manifest.schema_version} != {SCHEMA_VERSION}")
        if manifest.corpus_sha256 != expected_corpus_sha256:
            raise IndexLoadError(
                f"index {manifest.index_id} was built for corpus {manifest.corpus_sha256[:12]}, "
                f"but serving corpus {expected_corpus_sha256[:12]}; rebuild the index"
            )
        if matrix.dtype != np.float32 or matrix.shape != (len(ids), manifest.dim):
            raise IndexLoadError(
                f"embeddings {matrix.dtype}{matrix.shape} inconsistent with manifest"
            )
        if len(ids) != manifest.n_chunks:
            raise IndexLoadError("chunk_ids length != manifest.n_chunks")
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-3):
            raise IndexLoadError("embeddings are not L2-normalised")
        return cls(manifest, ids, matrix)

    def search(self, query_vec: Matrix, k: int) -> list[tuple[str, float]]:
        if query_vec.shape != (1, self.manifest.dim):
            raise ValueError(f"query shape {query_vec.shape} != (1, {self.manifest.dim})")
        scores = (self.matrix @ query_vec[0]).astype(np.float32)
        k = min(k, len(self.chunk_ids))
        top = np.argpartition(-scores, k - 1)[:k]
        # Stable tie-break on row index so results are deterministic.
        order = sorted(top.tolist(), key=lambda i: (-float(scores[i]), i))
        return [(self.chunk_ids[i], float(scores[i])) for i in order]


def resolve_index_dir(root: Path) -> Path:
    latest = root / "LATEST"
    if not latest.exists():
        raise IndexLoadError(f"{latest} not found; run `make index` first")
    return root / latest.read_text(encoding="utf-8").strip()


def main(argv: Sequence[str] | None = None) -> int:
    from ukmoney_rag.embed import FastEmbedEmbedder

    p = argparse.ArgumentParser(description="Build a versioned dense index from chunks.jsonl")
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("indexes"))
    p.add_argument("--model-cache", type=Path, default=Path(".cache/models"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    chunks, corpus_sha = load_chunks(args.data_dir / "chunks.jsonl")
    embedder = FastEmbedEmbedder(cache_dir=args.model_cache)
    path = build_index(chunks, corpus_sha, embedder, args.out)
    print(f"index_id={path.name} corpus_sha256={corpus_sha} n_chunks={len(chunks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

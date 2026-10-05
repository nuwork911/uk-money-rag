"""Dense index tests. Offline: a hashing FakeEmbedder stands in for the real model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fakes import CORPUS, FakeEmbedder, make_chunk
from ukmoney_rag.dense import (
    DenseIndex,
    IndexLoadError,
    build_index,
    compute_index_id,
    load_chunks,
    resolve_index_dir,
)
from ukmoney_rag.models import Chunk


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        make_chunk(0, "Income Tax rates", "higher rate income tax band 40 percent"),
        make_chunk(1, "State Pension age", "check when you reach state pension age"),
        make_chunk(2, "Universal Credit", "universal credit payments and eligibility"),
    ]


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


def test_build_then_load_roundtrip(
    tmp_path: Path, chunks: list[Chunk], embedder: FakeEmbedder
) -> None:
    path = build_index(chunks, CORPUS, embedder, tmp_path)
    assert resolve_index_dir(tmp_path) == path
    idx = DenseIndex.load(path, expected_corpus_sha256=CORPUS)
    assert idx.chunk_ids == [c.chunk_id for c in chunks]
    assert idx.manifest.n_chunks == 3
    assert idx.manifest.dim == 64
    assert not list(tmp_path.glob(".tmp-*")), "temp dir left behind"


def test_search_ranks_lexically_related_chunk_first(
    tmp_path: Path, chunks: list[Chunk], embedder: FakeEmbedder
) -> None:
    idx = DenseIndex.load(build_index(chunks, CORPUS, embedder, tmp_path), CORPUS)
    hits = idx.search(embedder.embed_query("when is my state pension age"), k=2)
    assert hits[0][0] == chunks[1].chunk_id
    assert len(hits) == 2
    assert hits[0][1] >= hits[1][1]


def test_search_k_larger_than_corpus(
    tmp_path: Path, chunks: list[Chunk], embedder: FakeEmbedder
) -> None:
    idx = DenseIndex.load(build_index(chunks, CORPUS, embedder, tmp_path), CORPUS)
    assert len(idx.search(embedder.embed_query("tax"), k=50)) == 3


def test_search_rejects_wrong_query_shape(
    tmp_path: Path, chunks: list[Chunk], embedder: FakeEmbedder
) -> None:
    idx = DenseIndex.load(build_index(chunks, CORPUS, embedder, tmp_path), CORPUS)
    with pytest.raises(ValueError, match="query shape"):
        idx.search(np.zeros((1, 8), dtype=np.float32), k=1)


def test_build_is_idempotent(tmp_path: Path, chunks: list[Chunk], embedder: FakeEmbedder) -> None:
    p1 = build_index(chunks, CORPUS, embedder, tmp_path)
    mtime = (p1 / "manifest.json").stat().st_mtime_ns
    p2 = build_index(chunks, CORPUS, embedder, tmp_path)
    assert p1 == p2
    assert (p2 / "manifest.json").stat().st_mtime_ns == mtime


@pytest.mark.parametrize(
    "change",
    [
        {"corpus_sha256": "b" * 64},
        {"model_id": "other/model"},
        {"model_sha256": "0" * 64},
        {"query_prefix": ""},
    ],
)
def test_index_id_changes_with_every_input(change: dict[str, str]) -> None:
    base = {
        "corpus_sha256": CORPUS,
        "model_id": "m",
        "model_sha256": "f" * 64,
        "query_prefix": "q: ",
    }
    assert compute_index_id(**base) != compute_index_id(**{**base, **change})


def test_load_refuses_index_for_different_corpus(
    tmp_path: Path, chunks: list[Chunk], embedder: FakeEmbedder
) -> None:
    path = build_index(chunks, CORPUS, embedder, tmp_path)
    with pytest.raises(IndexLoadError, match="rebuild the index"):
        DenseIndex.load(path, expected_corpus_sha256="b" * 64)


def test_load_detects_truncated_embeddings(
    tmp_path: Path, chunks: list[Chunk], embedder: FakeEmbedder
) -> None:
    path = build_index(chunks, CORPUS, embedder, tmp_path)
    np.save(path / "embeddings.npy", np.load(path / "embeddings.npy")[:2])
    with pytest.raises(IndexLoadError, match="inconsistent"):
        DenseIndex.load(path, CORPUS)


def test_load_detects_unnormalised_embeddings(
    tmp_path: Path, chunks: list[Chunk], embedder: FakeEmbedder
) -> None:
    path = build_index(chunks, CORPUS, embedder, tmp_path)
    np.save(path / "embeddings.npy", np.load(path / "embeddings.npy") * 2)
    with pytest.raises(IndexLoadError, match="normalised"):
        DenseIndex.load(path, CORPUS)


def test_load_missing_dir_is_clear_error(tmp_path: Path) -> None:
    with pytest.raises(IndexLoadError):
        DenseIndex.load(tmp_path / "nope", CORPUS)
    with pytest.raises(IndexLoadError, match="make index"):
        resolve_index_dir(tmp_path)


def test_duplicate_chunk_ids_rejected(
    tmp_path: Path, chunks: list[Chunk], embedder: FakeEmbedder
) -> None:
    with pytest.raises(IndexLoadError, match="duplicate"):
        build_index([chunks[0], chunks[0]], CORPUS, embedder, tmp_path)


def test_load_chunks_hashes_exact_bytes(tmp_path: Path, chunks: list[Chunk]) -> None:
    f = tmp_path / "chunks.jsonl"
    f.write_text("".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8")
    loaded, sha = load_chunks(f)
    assert loaded == chunks
    assert sha == hashlib.sha256(f.read_bytes()).hexdigest()
    f.write_text("", encoding="utf-8")
    with pytest.raises(IndexLoadError):
        load_chunks(f)


def test_manifest_is_human_readable_json(
    tmp_path: Path, chunks: list[Chunk], embedder: FakeEmbedder
) -> None:
    m = json.loads((build_index(chunks, CORPUS, embedder, tmp_path) / "manifest.json").read_text())
    assert m["corpus_sha256"] == CORPUS
    assert m["query_prefix"] == "q: "


def test_rebuilding_an_older_corpus_repoints_latest(
    tmp_path: Path, chunks: list[Chunk], embedder: FakeEmbedder
) -> None:
    old = build_index(chunks, CORPUS, embedder, tmp_path)
    build_index(chunks, "b" * 64, embedder, tmp_path)
    assert build_index(chunks, CORPUS, embedder, tmp_path) == old
    assert resolve_index_dir(tmp_path) == old


def test_cli_builds_index_offline(
    tmp_path: Path,
    chunks: list[Chunk],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import ukmoney_rag.embed as embed_mod
    from ukmoney_rag.dense import main

    monkeypatch.setattr(embed_mod, "FastEmbedEmbedder", lambda cache_dir: FakeEmbedder())
    data = tmp_path / "processed"
    data.mkdir()
    (data / "chunks.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in chunks))
    assert main(["--data-dir", str(data), "--out", str(tmp_path / "idx")]) == 0
    out = capsys.readouterr().out
    assert f"index_id={resolve_index_dir(tmp_path / 'idx').name}" in out

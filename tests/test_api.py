from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakes import FakeEmbedder
from ukmoney_rag import api_dense
from ukmoney_rag.api import create_app
from ukmoney_rag.api_dense import DenseRetriever
from ukmoney_rag.dense import (
    DenseIndex,
    IndexLoadError,
    build_index,
    load_chunks,
    resolve_index_dir,
)


@pytest.fixture
def client(processed_dir: Path) -> TestClient:
    return TestClient(create_app(processed_dir))


@pytest.fixture
def dense_index_root(processed_dir: Path, tmp_path: Path) -> Path:
    """A real dense index over the fixture corpus, built with the offline FakeEmbedder."""
    chunks, sha = load_chunks(processed_dir / "chunks.jsonl")
    root = tmp_path / "indexes"
    build_index(chunks, sha, FakeEmbedder(), root)
    return root


@pytest.fixture
def dense_client(processed_dir: Path, dense_index_root: Path) -> TestClient:
    _, sha = load_chunks(processed_dir / "chunks.jsonl")
    retriever = DenseRetriever(
        DenseIndex.load(resolve_index_dir(dense_index_root), sha), FakeEmbedder()
    )
    return TestClient(create_app(processed_dir, dense=retriever))


def test_health_reports_corpus_provenance(client: TestClient, processed_dir: Path) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["n_chunks"] > 0
    _, sha = load_chunks(processed_dir / "chunks.jsonl")
    assert body["corpus_sha256"] == sha  # computed from the served bytes, not copied
    assert body["retrievers"] == ["bm25"]
    assert body["dense_index_id"] is None
    assert body["embedding_model"] is None


def test_search_returns_cited_results_and_disclaimer(client: TestClient) -> None:
    response = client.get("/search", params={"q": "how much is the weekly rate", "k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "bm25"  # default mode
    assert 1 <= len(body["results"]) <= 3
    assert body["results"][0]["url"].startswith("https://www.gov.uk/sample-guide")
    assert "not financial advice" in body["disclaimer"]


@pytest.mark.parametrize(
    "params",
    [{"q": "x"}, {"q": "pension", "k": 50}, {}, {"q": "pension", "mode": "hybrid"}],
)
def test_search_rejects_invalid_input(client: TestClient, params: dict[str, object]) -> None:
    assert client.get("/search", params=params).status_code == 422


def test_dense_mode_unavailable_is_503_not_silent_fallback(client: TestClient) -> None:
    response = client.get("/search", params={"q": "weekly rate", "mode": "dense"})
    assert response.status_code == 503
    assert "not enabled" in response.json()["detail"]


def test_dense_search_returns_cited_results(dense_client: TestClient) -> None:
    response = dense_client.get("/search", params={"q": "weekly rate", "k": 2, "mode": "dense"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "dense"
    assert len(body["results"]) == 2
    assert all(r["url"].startswith("https://") and r["text"] for r in body["results"])
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_health_reports_dense_index_when_enabled(dense_client: TestClient) -> None:
    body = dense_client.get("/health").json()
    assert body["retrievers"] == ["bm25", "dense"]
    assert body["embedding_model"] == FakeEmbedder.model_id
    assert len(body["dense_index_id"]) == 12


def test_env_flag_enables_dense(
    processed_dir: Path, dense_index_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UKMONEY_DENSE", "1")
    monkeypatch.setenv("UKMONEY_INDEX_ROOT", str(dense_index_root))
    monkeypatch.setattr(api_dense, "FastEmbedEmbedder", lambda cache_dir: FakeEmbedder())
    body = TestClient(create_app(processed_dir)).get("/health").json()
    assert body["retrievers"] == ["bm25", "dense"]


def test_env_flag_with_missing_index_fails_fast(
    processed_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UKMONEY_DENSE", "1")
    monkeypatch.setenv("UKMONEY_INDEX_ROOT", str(tmp_path / "no-indexes"))

    def _no_model(cache_dir: Path) -> FakeEmbedder:
        raise AssertionError("model must not load before the index is validated")

    monkeypatch.setattr(api_dense, "FastEmbedEmbedder", _no_model)
    with pytest.raises(IndexLoadError, match="make index"):
        create_app(processed_dir)


def test_dense_index_for_other_corpus_is_refused(
    processed_dir: Path, tmp_path: Path, dense_index_root: Path
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    lines = (processed_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    (other / "chunks.jsonl").write_text("\n".join(lines[:1]) + "\n", encoding="utf-8")
    _, sha = load_chunks(processed_dir / "chunks.jsonl")
    retriever = DenseRetriever(
        DenseIndex.load(resolve_index_dir(dense_index_root), sha), FakeEmbedder()
    )
    with pytest.raises(IndexLoadError, match="different corpus"):
        create_app(other, dense=retriever)


def test_manifest_out_of_sync_with_corpus_fails_fast(processed_dir: Path) -> None:
    manifest = processed_dir / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["chunks_sha256"] = "0" * 64
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(IndexLoadError, match="out of sync"):
        create_app(processed_dir)


def test_root_redirects_to_docs(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_missing_corpus_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make ingest"):
        create_app(tmp_path)


def test_updated_at_is_documented_as_major_update_only(client: TestClient) -> None:
    """Regression test for known failure coarse-updated-at (eval/known_failures.toml)."""
    schema = client.get("/openapi.json").json()["components"]["schemas"]["SearchResult"]
    description = schema["properties"]["updated_at"]["description"]
    assert "*major*" in description
    assert "not a freshness guarantee" in description

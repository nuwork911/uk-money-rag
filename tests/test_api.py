from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ukmoney_rag.api import create_app


@pytest.fixture
def client(processed_dir: Path) -> TestClient:
    return TestClient(create_app(processed_dir))


def test_health_reports_corpus_provenance(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["n_chunks"] > 0
    assert len(body["corpus_sha256"]) == 64


def test_search_returns_cited_results_and_disclaimer(client: TestClient) -> None:
    response = client.get("/search", params={"q": "how much is the weekly rate", "k": 3})
    assert response.status_code == 200
    body = response.json()
    assert 1 <= len(body["results"]) <= 3
    assert body["results"][0]["url"].startswith("https://www.gov.uk/sample-guide")
    assert "not financial advice" in body["disclaimer"]


@pytest.mark.parametrize("params", [{"q": "x"}, {"q": "pension", "k": 50}, {}])
def test_search_rejects_invalid_input(client: TestClient, params: dict[str, object]) -> None:
    assert client.get("/search", params=params).status_code == 422


def test_root_redirects_to_docs(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_missing_corpus_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make ingest"):
        create_app(tmp_path)

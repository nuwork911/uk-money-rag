from __future__ import annotations

from pathlib import Path

import pytest

from fakes import CORPUS, FakeEmbedder, make_chunk
from ukmoney_rag.api_dense import DenseRetriever
from ukmoney_rag.dense import IndexLoadError, build_index


@pytest.fixture
def index_root(tmp_path: Path) -> Path:
    chunks = [
        make_chunk(0, "Income Tax rates", "higher rate income tax"),
        make_chunk(1, "State Pension age", "state pension age"),
    ]
    build_index(chunks, CORPUS, FakeEmbedder(), tmp_path)
    return tmp_path


def test_from_env_loads_and_searches(index_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UKMONEY_INDEX_ROOT", str(index_root))
    r = DenseRetriever.from_env(CORPUS, embedder=FakeEmbedder())
    assert r.search("pension age", k=1)[0][0] == "doc1:000000000001"


def test_refuses_embedder_that_did_not_build_the_index(
    index_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UKMONEY_INDEX_ROOT", str(index_root))
    with pytest.raises(IndexLoadError, match="built with"):
        DenseRetriever.from_env(CORPUS, embedder=FakeEmbedder(query_prefix=""))

"""Real-model tests. Slow (downloads ~65 MB once); run with: pytest -m model"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ukmoney_rag.embed import BGE_QUERY_PREFIX, FastEmbedEmbedder

pytestmark = pytest.mark.model

CACHE = Path(".cache/models")


@pytest.fixture(scope="module")
def real() -> FastEmbedEmbedder:
    return FastEmbedEmbedder(cache_dir=CACHE)


def test_shape_norm_and_pinned_hash(real: FastEmbedEmbedder) -> None:
    v = real.embed_query("higher rate tax")
    assert v.shape == (1, 384)
    assert v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-4
    assert real.query_prefix == BGE_QUERY_PREFIX


def test_paraphrase_beats_unrelated(real: FastEmbedEmbedder) -> None:
    """The point of dense retrieval: no shared keywords, still matches."""
    docs = real.embed_passages(
        [
            (
                "State Pension age > Overview\n\nYou can get the State Pension when you reach "
                "State Pension age."
            ),
            "Income Tax rates > Current rates\n\nThe higher rate is 40%.",
        ]
    )
    q = real.embed_query("when am I old enough to claim retirement money from the government?")
    scores = docs @ q[0]
    assert scores[0] > scores[1]


def test_wrong_pinned_hash_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="upstream model changed"):
        FastEmbedEmbedder(cache_dir=CACHE, expected_sha256="0" * 64)

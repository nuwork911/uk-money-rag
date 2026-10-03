"""Smoke test against the real GOV.UK Content API. Run with: make test-live"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ukmoney_rag.chunk import chunk_document
from ukmoney_rag.fetch import Fetcher
from ukmoney_rag.models import Source
from ukmoney_rag.parse import parse_document

pytestmark = pytest.mark.network


def test_real_govuk_guide_parses_and_chunks(tmp_path: Path) -> None:
    source = Source(
        name="live-state-pension",
        kind="govuk_api",
        url="https://www.gov.uk/new-state-pension",
        licence="OGL-v3.0",
    )
    fetcher = Fetcher(tmp_path)
    try:
        fetched = fetcher.fetch(source)
    finally:
        fetcher.close()
    doc = parse_document(source, fetched.content, datetime.now(UTC))
    chunks = chunk_document(doc)

    assert len(doc.sections) >= 3
    assert doc.updated_at is not None
    assert len(chunks) >= len(doc.sections)
    assert all(c.url.startswith("https://www.gov.uk/new-state-pension") for c in chunks)

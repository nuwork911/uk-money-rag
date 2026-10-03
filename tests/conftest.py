from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ukmoney_rag.models import Source

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 9, 29, 12, 0, tzinfo=UTC)


@pytest.fixture
def govuk_source() -> Source:
    return Source(
        name="sample-guide",
        kind="govuk_api",
        url="https://www.gov.uk/sample-guide",
        licence="OGL-v3.0",
    )


@pytest.fixture
def html_source() -> Source:
    return Source(
        name="sample-page",
        kind="html",
        url="https://www.example.org/en/sample-page",
        licence="test-licence",
    )


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    """A real corpus built by the ingest pipeline (offline) from the test fixtures."""
    import shutil

    from ukmoney_rag.ingest import main

    cache = tmp_path / "raw"
    cache.mkdir()
    shutil.copy(FIXTURES / "govuk_guide.json", cache / "sample-guide.json")
    shutil.copy(FIXTURES / "sample_page.html", cache / "sample-page.html")
    config = tmp_path / "sources.toml"
    config.write_text(
        '[[source]]\nname = "sample-guide"\nkind = "govuk_api"\n'
        'url = "https://www.gov.uk/sample-guide"\nlicence = "OGL-v3.0"\n\n'
        '[[source]]\nname = "sample-page"\nkind = "html"\n'
        'url = "https://www.example.org/en/sample-page"\nlicence = "test-licence"\n'
    )
    out = tmp_path / "processed"
    code = main(
        ["--sources", str(config), "--cache-dir", str(cache), "--out-dir", str(out), "--offline"]
    )
    assert code == 0
    return out

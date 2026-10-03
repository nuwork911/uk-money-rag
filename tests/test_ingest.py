from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ukmoney_rag.ingest import load_sources, main

CONFIG = """
[[source]]
name = "sample-guide"
kind = "govuk_api"
url = "https://www.gov.uk/sample-guide"
licence = "OGL-v3.0"

[[source]]
name = "sample-page"
kind = "html"
url = "https://www.example.org/en/sample-page"
licence = "test-licence"

[[source]]
name = "blocked-optional"
kind = "html"
url = "https://www.example.org/blocked"
licence = "unverified"
optional = true
"""


def _setup(tmp_path: Path, fixtures_dir: Path) -> tuple[Path, Path]:
    cache = tmp_path / "raw"
    cache.mkdir()
    shutil.copy(fixtures_dir / "govuk_guide.json", cache / "sample-guide.json")
    shutil.copy(fixtures_dir / "sample_page.html", cache / "sample-page.html")
    config = tmp_path / "sources.toml"
    config.write_text(CONFIG)
    return config, cache


def _run(config: Path, cache: Path, out: Path) -> int:
    return main(["--sources", str(config), "--cache-dir", str(cache), "--out-dir", str(out),
                 "--offline"])  # fmt: skip


def test_end_to_end_offline_and_reproducible(tmp_path: Path, fixtures_dir: Path) -> None:
    config, cache = _setup(tmp_path, fixtures_dir)

    assert _run(config, cache, tmp_path / "out1") == 0  # optional failure does not fail the run
    assert _run(config, cache, tmp_path / "out2") == 0

    manifest = json.loads((tmp_path / "out1" / "manifest.json").read_text())
    assert manifest["n_documents"] == 2
    assert set(manifest["failures"]) == {"blocked-optional"}
    assert manifest["n_chunks"] == len(
        (tmp_path / "out1" / "chunks.jsonl").read_text().splitlines()
    )

    # Same cached inputs + same config => byte-identical chunks file.
    assert (tmp_path / "out1" / "chunks.jsonl").read_bytes() == (
        tmp_path / "out2" / "chunks.jsonl"
    ).read_bytes()

    first = json.loads((tmp_path / "out1" / "chunks.jsonl").read_text().splitlines()[0])
    assert {"chunk_id", "url", "section_title", "text", "licence", "updated_at"} <= first.keys()


def test_required_source_failure_returns_nonzero(tmp_path: Path, fixtures_dir: Path) -> None:
    config, cache = _setup(tmp_path, fixtures_dir)
    (cache / "sample-guide.json").unlink()  # required source now missing
    assert _run(config, cache, tmp_path / "out") == 1


def test_load_sources_rejects_duplicate_names(tmp_path: Path) -> None:
    config = tmp_path / "dup.toml"
    block = '[[source]]\nname = "a"\nkind = "html"\nurl = "https://x.test"\nlicence = "l"\n'
    config.write_text(block * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_sources(config)

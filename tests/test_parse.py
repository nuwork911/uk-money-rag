from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ukmoney_rag.models import Source
from ukmoney_rag.parse import (
    ParseError,
    html_to_blocks,
    latest_update,
    parse_govuk,
    parse_html,
)


def test_html_to_blocks_preserves_structure() -> None:
    html = (
        "<h2>Rates</h2><p>Pay is <span>&pound;10</span>.</p>"
        "<ul><li>one</li><li>two</li></ul>"
        "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        "<script>evil()</script><!-- hidden -->"
    )
    assert html_to_blocks(html) == [
        "## Rates",
        "Pay is £10.",
        "- one\n- two",
        "| A | B |\n| 1 | 2 |",
    ]


def test_parse_govuk_guide(govuk_source: Source, fixtures_dir: Path, now: datetime) -> None:
    raw = (fixtures_dir / "govuk_guide.json").read_text()
    doc = parse_govuk(raw, govuk_source, now)

    assert doc.title == "Sample benefit guide"
    assert [s.slug for s in doc.sections] == ["overview", "how-much-you-get"]  # empty part dropped
    assert doc.sections[1].url == "https://www.gov.uk/sample-guide/how-much-you-get"
    assert "£184.90" in doc.sections[1].text
    assert "editorial comment" not in doc.sections[1].text
    assert doc.licence == "OGL-v3.0"


def test_latest_update_prefers_change_history_over_stale_public_updated_at() -> None:
    payload = {
        "public_updated_at": "2015-03-17T18:23:25+00:00",
        "details": {"change_history": [{"public_timestamp": "2026-04-05T23:01:01Z"}]},
    }
    assert latest_update(payload) == datetime(2026, 4, 5, 23, 1, 1, tzinfo=UTC)
    assert latest_update({}) is None


def test_parse_govuk_rejects_withdrawn_page(govuk_source: Source, now: datetime) -> None:
    raw = json.dumps({"title": "x", "withdrawn_notice": {"explanation": "gone"}, "details": {}})
    with pytest.raises(ParseError, match="withdrawn"):
        parse_govuk(raw, govuk_source, now)


def test_parse_govuk_rejects_unsupported_schema(govuk_source: Source, now: datetime) -> None:
    raw = json.dumps({"title": "x", "schema_name": "transaction", "details": {}})
    with pytest.raises(ParseError, match=r"unsupported GOV\.UK schema"):
        parse_govuk(raw, govuk_source, now)


def test_parse_html_strips_boilerplate_and_splits_on_h2(
    html_source: Source, fixtures_dir: Path, now: datetime
) -> None:
    doc = parse_html((fixtures_dir / "sample_page.html").read_text(), html_source, now)

    assert doc.title == "Beginner budgeting guide"
    assert doc.description == "A synthetic page used to test the generic HTML parser."
    assert [s.title for s in doc.sections] == [
        "Beginner budgeting guide",
        "Work out your income",
        "List your spending",
    ]
    all_text = " ".join(s.text for s in doc.sections)
    for junk in ("SHOULD NOT APPEAR", "Accept cookies", "unrelated footer", "Home"):
        assert junk not in all_text
    assert "- Wages after tax" in doc.sections[1].text
    assert "| Household | Rent, energy |" in doc.sections[2].text


def test_parse_html_rejects_near_empty_page(html_source: Source, now: datetime) -> None:
    with pytest.raises(ParseError, match=r"only \d+ words extracted"):
        parse_html(
            "<html><body><main><p>Please enable JavaScript.</p></main></body></html>",
            html_source,
            now,
        )

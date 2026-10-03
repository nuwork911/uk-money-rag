"""Raw payload -> typed Document.

GOV.UK comes from the Content API (structured JSON, HTML fragments inside); other sites are
parsed generically from <main>/<article>. Both routes share one HTML->text converter that keeps
headings, lists and tables, because chunk quality depends on structure surviving.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString, PreformattedString

from ukmoney_rag.models import Document, Section, Source

MIN_DOC_WORDS = 50  # below this, assume a wrong selector / blocked page rather than real content

_HEADINGS = {f"h{i}": i for i in range(1, 7)}
_INLINE = frozenset(
    {"a", "span", "strong", "b", "em", "i", "u", "abbr", "cite", "code", "sup", "sub", "small",
     "mark", "time", "label"}
)  # fmt: skip
_TEXT_BLOCKS = frozenset({"p", "blockquote", "pre", "dt", "dd", "figcaption"})
_NOISE = (
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "aside",
    "noscript",
    "svg",
    "iframe",
    "button",
)
_WS = re.compile(r"\s+")


class ParseError(Exception):
    """Payload was fetched but could not be turned into a usable Document."""


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def _strip_noise(node: Tag) -> None:
    for tag in node.find_all(_NOISE):
        tag.decompose()
    for br in node.find_all("br"):
        br.replace_with(" ")


def blocks_from_node(node: Tag) -> list[str]:
    """Flatten a DOM subtree into text blocks.

    Headings become '## Title'; a list or table is ONE multi-line block so the chunker keeps it
    together; everything else is one paragraph per block.
    """
    blocks: list[str] = []
    inline: list[str] = []

    def flush() -> None:
        text = _clean("".join(inline))
        inline.clear()
        if text:
            blocks.append(text)

    for child in node.children:
        if isinstance(child, PreformattedString):  # comments, doctype, CDATA
            continue
        if isinstance(child, NavigableString):
            inline.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name in _INLINE:
            inline.append(child.get_text())
            continue
        flush()
        if name in _HEADINGS:
            text = _clean(child.get_text())
            if text:
                blocks.append(f"{'#' * _HEADINGS[name]} {text}")
        elif name in _TEXT_BLOCKS:
            text = _clean(child.get_text())
            if text:
                blocks.append(text)
        elif name in ("ul", "ol"):
            items = [_clean(li.get_text()) for li in child.find_all("li", recursive=False)]
            lines = [f"- {item}" for item in items if item]
            if lines:
                blocks.append("\n".join(lines))
        elif name == "table":
            rows = []
            for tr in child.find_all("tr"):
                cells = [_clean(c.get_text()) for c in tr.find_all(["th", "td"])]
                if any(cells):
                    rows.append("| " + " | ".join(cells) + " |")
            if rows:
                blocks.append("\n".join(rows))
        else:  # div, section, article, ... -> recurse
            blocks.extend(blocks_from_node(child))
    flush()
    return blocks


def html_to_blocks(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    _strip_noise(soup)
    return blocks_from_node(soup)


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def latest_update(payload: dict[str, Any]) -> datetime | None:
    """Most recent of public_updated_at and change_history entries.

    GOV.UK's public_updated_at can lag the real content (e.g. rate changes recorded only in
    change_history), so freshness metadata is the max of both.
    """
    candidates = [payload.get("public_updated_at")]
    details = payload.get("details") or {}
    for entry in details.get("change_history") or []:
        candidates.append(entry.get("public_timestamp"))
    parsed = [dt for c in candidates if (dt := _parse_dt(c)) is not None]
    return max(parsed) if parsed else None


def _finalise(doc: Document) -> Document:
    if not doc.sections or doc.n_words < MIN_DOC_WORDS:
        raise ParseError(
            f"{doc.source_name}: only {doc.n_words} words extracted "
            f"(minimum {MIN_DOC_WORDS}); wrong selector, JS-rendered page, or blocked?"
        )
    return doc


def parse_govuk(raw: str, source: Source, fetched_at: datetime) -> Document:
    try:
        payload: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParseError(f"{source.name}: invalid JSON") from exc
    if payload.get("withdrawn_notice"):
        raise ParseError(f"{source.name}: page has been withdrawn")

    details = payload.get("details") or {}
    base_url = source.url.rstrip("/")
    sections: list[Section] = []

    if details.get("parts"):  # multi-part "guide"
        for part in details["parts"]:
            text = "\n\n".join(html_to_blocks(part.get("body", "")))
            if text:
                sections.append(
                    Section(
                        title=part["title"],
                        slug=part["slug"],
                        url=f"{base_url}/{part['slug']}",
                        text=text,
                    )
                )
    elif details.get("body"):  # single-body formats (e.g. "answer")
        text = "\n\n".join(html_to_blocks(details["body"]))
        if text:
            sections.append(Section(title=payload["title"], slug="main", url=base_url, text=text))
    else:
        schema = payload.get("schema_name", "unknown")
        raise ParseError(f"{source.name}: unsupported GOV.UK schema '{schema}' (no parts/body)")

    return _finalise(
        Document(
            doc_id=source.name,
            source_name=source.name,
            url=base_url,
            title=payload["title"],
            description=payload.get("description"),
            licence=source.licence,
            updated_at=latest_update(payload),
            fetched_at=fetched_at,
            sections=tuple(sections),
        )
    )


def _page_title(soup: BeautifulSoup, fallback: str) -> str:
    h1 = soup.find("h1")
    if isinstance(h1, Tag) and _clean(h1.get_text()):
        return _clean(h1.get_text())
    if soup.title is not None and soup.title.string:
        return _clean(soup.title.string)
    return fallback


def _meta_description(soup: BeautifulSoup) -> str | None:
    meta = soup.find("meta", attrs={"name": "description"})
    if isinstance(meta, Tag):
        content = meta.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def parse_html(raw: str, source: Source, fetched_at: datetime) -> Document:
    soup = BeautifulSoup(raw, "html.parser")
    title = _page_title(soup, fallback=source.name)
    description = _meta_description(soup)

    main = soup.find("main") or soup.find("article") or soup.body
    if not isinstance(main, Tag):
        raise ParseError(f"{source.name}: no <main>, <article> or <body> found")
    _strip_noise(main)

    sections: list[Section] = []
    current_title = title
    current: list[str] = []

    def emit() -> None:
        text = "\n\n".join(current)
        if text:
            slug = f"{len(sections):02d}-{_slugify(current_title)}"
            sections.append(Section(title=current_title, slug=slug, url=source.url, text=text))

    for block in blocks_from_node(main):
        if block == f"# {title}":
            continue  # the page title is metadata, not content
        if block.startswith("## "):
            emit()
            current_title, current = block[3:], []
        else:
            current.append(block)
    emit()

    return _finalise(
        Document(
            doc_id=source.name,
            source_name=source.name,
            url=source.url,
            title=title,
            description=description,
            licence=source.licence,
            updated_at=None,  # generic HTML has no reliable last-updated signal
            fetched_at=fetched_at,
            sections=tuple(sections),
        )
    )


def parse_document(source: Source, raw: str, fetched_at: datetime) -> Document:
    if source.kind == "govuk_api":
        return parse_govuk(raw, source, fetched_at)
    return parse_html(raw, source, fetched_at)

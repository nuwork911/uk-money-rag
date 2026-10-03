"""Structure-aware chunking.

Strategy (deliberately simple and inspectable for Week 1):
1. Chunk within sections only, so a chunk never straddles two headings' worth of topic.
2. Treat paragraphs / lists / tables as atomic units; glue headings and lead-in sentences
   (ending in ':') to the unit after them, so a list is never separated from its intro.
3. Greedily pack units up to `max_words`; carry whole trailing units forward as overlap.
4. Only a unit that alone exceeds `max_words` is split (by line, then by word window).

`words` is a stand-in for tokens. In Week 2, swap `_wc` for your embedding model's tokenizer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ukmoney_rag.models import Chunk, Document

_PARA_SPLIT = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class ChunkConfig:
    max_words: int = 180
    overlap_words: int = 30

    def __post_init__(self) -> None:
        if self.max_words < 1:
            raise ValueError("max_words must be >= 1")
        if not 0 <= self.overlap_words < self.max_words:
            raise ValueError("overlap_words must satisfy 0 <= overlap_words < max_words")


def _wc(text: str) -> int:
    return len(text.split())


def _is_heading(unit: str) -> bool:
    return unit.startswith("#") and "\n" not in unit


def _is_lead_in(unit: str) -> bool:
    """A sentence introducing what follows ("You must tell HMRC if you:") is meaningless alone."""
    return unit.endswith(":") and "\n" not in unit


def _glue_leads(paragraphs: list[str]) -> list[str]:
    """Attach headings and lead-in sentences to the unit that follows them.

    A trailing orphan heading is dropped (it introduces nothing); a trailing lead-in is kept.
    """
    units: list[str] = []
    pending: list[str] = []
    for para in paragraphs:
        if _is_heading(para) or _is_lead_in(para):
            pending.append(para)
        else:
            units.append("\n\n".join([*pending, para]))
            pending = []
    leftover = [p for p in pending if not _is_heading(p)]
    if leftover:
        units.append("\n\n".join(leftover))
    return units


def _explode(unit: str, max_words: int) -> list[str]:
    """Split an oversized unit: by line first (keeps list items/table rows whole), then by word."""
    if _wc(unit) <= max_words:
        return [unit]
    lines = [ln for ln in unit.split("\n") if ln.strip()]
    if len(lines) > 1:
        return [piece for line in lines for piece in _explode(line, max_words)]
    words = unit.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _overlap_tail(units: list[str], limit: int) -> list[str]:
    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        words = _wc(unit)
        if total + words > limit:
            break
        tail.append(unit)
        total += words
    tail.reverse()
    return tail


def split_text(text: str, cfg: ChunkConfig) -> list[str]:
    paragraphs = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    units = [piece for unit in _glue_leads(paragraphs) for piece in _explode(unit, cfg.max_words)]

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        size = _wc(unit)
        if current and sum(map(_wc, current)) + size > cfg.max_words:
            chunks.append("\n\n".join(current))
            current = _overlap_tail(current, cfg.overlap_words)
            if current and sum(map(_wc, current)) + size > cfg.max_words:
                current = []  # overlap would push us over the limit; drop it
        current.append(unit)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _chunk_id(doc_id: str, section_slug: str, text: str) -> str:
    """Content-addressed: unchanged text keeps its ID, so Week 2 can skip re-embedding it."""
    digest = hashlib.sha256(f"{doc_id}\x1f{section_slug}\x1f{text}".encode()).hexdigest()
    return f"{doc_id}:{digest[:12]}"


def chunk_document(doc: Document, cfg: ChunkConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkConfig()
    chunks: list[Chunk] = []
    for section in doc.sections:
        for text in split_text(section.text, cfg):
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(doc.doc_id, section.slug, text),
                    doc_id=doc.doc_id,
                    chunk_index=len(chunks),
                    url=section.url,
                    title=doc.title,
                    section_title=section.title,
                    text=text,
                    n_words=_wc(text),
                    licence=doc.licence,
                    updated_at=doc.updated_at,
                )
            )
    return chunks

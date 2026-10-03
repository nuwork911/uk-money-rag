"""Typed, immutable data contracts shared by every pipeline stage."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

SourceKind = Literal["govuk_api", "html"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Source(_Frozen):
    """One entry in configs/sources.toml."""

    name: str  # unique slug; doubles as doc_id and cache filename
    kind: SourceKind
    url: str  # human-facing page URL (used for citations)
    licence: str
    optional: bool = False  # optional sources warn on failure instead of failing the run


class Section(_Frozen):
    title: str
    slug: str
    url: str  # deep link where possible (GOV.UK guide parts have their own URLs)
    text: str


class Document(_Frozen):
    doc_id: str
    source_name: str
    url: str
    title: str
    description: str | None
    licence: str
    updated_at: datetime | None
    fetched_at: datetime
    sections: tuple[Section, ...]

    @property
    def n_words(self) -> int:
        return sum(len(s.text.split()) for s in self.sections)


class Chunk(_Frozen):
    """Unit of retrieval. Deliberately excludes fetched_at so output is byte-reproducible."""

    chunk_id: str
    doc_id: str
    chunk_index: int  # position within the document
    url: str
    title: str
    section_title: str
    text: str
    n_words: int
    licence: str
    updated_at: datetime | None

    @property
    def embed_text(self) -> str:
        """Text to embed: the chunk plus its provenance, which helps short chunks retrieve well."""
        return f"{self.title} > {self.section_title}\n\n{self.text}"

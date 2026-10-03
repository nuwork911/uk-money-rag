from __future__ import annotations

import string
from datetime import datetime
from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ukmoney_rag.chunk import ChunkConfig, chunk_document, split_text
from ukmoney_rag.models import Document, Section


def _words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def _doc(now: datetime, *sections: Section) -> Document:
    return Document(
        doc_id="doc", source_name="doc", url="https://x.test/doc", title="Doc Title",
        description=None, licence="L", updated_at=None, fetched_at=now, sections=sections,
    )  # fmt: skip


def test_short_text_yields_single_chunk() -> None:
    assert split_text("Just one short paragraph.", ChunkConfig()) == ["Just one short paragraph."]


def test_chunks_never_exceed_max_words() -> None:
    text = "\n\n".join(_words(40, f"p{i}_") for i in range(10))
    cfg = ChunkConfig(max_words=100, overlap_words=20)
    chunks = split_text(text, cfg)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 100 for c in chunks)


def test_consecutive_chunks_overlap_on_whole_units() -> None:
    paras = [_words(30, f"p{i}_") for i in range(6)]
    chunks = split_text("\n\n".join(paras), ChunkConfig(max_words=70, overlap_words=30))
    for prev, nxt in pairwise(chunks):
        assert prev.split("\n\n")[-1] in nxt  # last paragraph of prev is repeated in next


def test_heading_stays_with_following_paragraph() -> None:
    text = f"{_words(60, 'a')}\n\n## Key heading\n\n{_words(60, 'b')}"
    chunks = split_text(text, ChunkConfig(max_words=70, overlap_words=0))
    heading_chunks = [c for c in chunks if "## Key heading" in c]
    assert len(heading_chunks) == 1
    assert "b0" in heading_chunks[0]  # heading is not orphaned at the end of a chunk


def test_oversized_list_splits_between_items_not_mid_item() -> None:
    items = [f"- {_words(15, f'i{n}_')}" for n in range(12)]
    chunks = split_text("\n".join(items), ChunkConfig(max_words=50, overlap_words=0))
    rebuilt = [line for c in chunks for line in c.split("\n") if line.strip()]
    assert rebuilt == items


def test_chunk_ids_are_deterministic_and_content_addressed(now: datetime) -> None:
    section = Section(title="S", slug="s", url="https://x.test/doc/s", text=_words(50))
    first = chunk_document(_doc(now, section))
    again = chunk_document(_doc(now, section))
    edited = chunk_document(_doc(now, section.model_copy(update={"text": _words(50) + " extra"})))

    assert [c.chunk_id for c in first] == [c.chunk_id for c in again]
    assert first[0].chunk_id != edited[0].chunk_id
    assert first[0].url == "https://x.test/doc/s"
    assert first[0].embed_text.startswith("Doc Title > S\n\n")


def test_chunks_do_not_cross_section_boundaries(now: datetime) -> None:
    a = Section(title="A", slug="a", url="u", text=_words(10, "a"))
    b = Section(title="B", slug="b", url="u", text=_words(10, "b"))
    chunks = chunk_document(_doc(now, a, b))
    assert [c.section_title for c in chunks] == ["A", "B"]


@pytest.mark.parametrize(("max_words", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_config_validation(max_words: int, overlap: int) -> None:
    with pytest.raises(ValueError, match=r"max_words|overlap_words"):
        ChunkConfig(max_words=max_words, overlap_words=overlap)


_word = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=8)
_paragraph = st.lists(_word, min_size=1, max_size=120).map(" ".join)


@given(
    paragraphs=st.lists(_paragraph, min_size=1, max_size=12),
    max_words=st.integers(min_value=5, max_value=80),
    data=st.data(),
)
def test_property_bounded_and_lossless(
    paragraphs: list[str], max_words: int, data: st.DataObject
) -> None:
    overlap = data.draw(st.integers(min_value=0, max_value=max_words - 1))
    chunks = split_text("\n\n".join(paragraphs), ChunkConfig(max_words, overlap))

    assert chunks
    assert all(len(c.split()) <= max_words for c in chunks)
    source_words = {w for p in paragraphs for w in p.split()}
    assert source_words <= {w for c in chunks for w in c.split()}  # nothing dropped


def test_lead_in_sentence_stays_with_its_list() -> None:
    lead = "You must tell HMRC if you:"
    text = f"{_words(60, 'a')}\n\n{lead}\n\n- change your name\n- move house"
    chunks = split_text(text, ChunkConfig(max_words=70, overlap_words=0))
    with_list = next(c for c in chunks if "- change your name" in c)
    assert lead in with_list


def test_trailing_lead_in_is_kept_but_trailing_heading_is_dropped() -> None:
    assert split_text("Body text.\n\nSee also:", ChunkConfig()) == ["Body text.\n\nSee also:"]
    assert split_text("Body text.\n\n## Empty heading", ChunkConfig()) == ["Body text."]

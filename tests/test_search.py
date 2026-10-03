from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ukmoney_rag.models import Chunk
from ukmoney_rag.search import BM25Index, tokenize


def _chunk(cid: str, text: str, section: str = "S") -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id="d", chunk_index=0, url=f"https://x.test/{cid}", title="T",
        section_title=section, text=text, n_words=len(text.split()), licence="L",
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )  # fmt: skip


def test_tokenize_normalises_amounts_and_drops_stopwords() -> None:
    assert tokenize("What is the £12,570 Personal Allowance?") == ["12570", "personal", "allowance"]
    assert tokenize("£184.90 a week") == ["184.90", "week"]


def test_relevant_chunk_ranks_first() -> None:
    index = BM25Index(
        [
            _chunk("tax", "Income Tax bands: basic rate 20%, higher rate 40%."),
            _chunk("pension", "The State Pension is paid every 4 weeks."),
            _chunk("uc", "Universal Credit is a payment to help with living costs."),
        ]
    )
    hits = index.search("higher rate income tax", k=2)
    assert [h.chunk.chunk_id for h in hits] == ["tax"]  # only chunks that match at all
    assert hits[0].score > 0


def test_stopword_only_query_returns_nothing() -> None:
    assert BM25Index([_chunk("a", "anything")]).search("what is the") == []


def test_ties_break_deterministically() -> None:
    index = BM25Index([_chunk("first", "pension credit"), _chunk("second", "pension credit")])
    assert [h.chunk.chunk_id for h in index.search("pension")] == ["first", "second"]


def test_loads_pipeline_output(processed_dir: Path) -> None:
    index = BM25Index.from_jsonl(processed_dir / "chunks.jsonl")
    assert len(index) > 0
    top = index.search("weekly rate")[0]
    assert top.chunk.url == "https://www.gov.uk/sample-guide/how-much-you-get"

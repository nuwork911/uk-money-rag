"""Lexical BM25 retrieval over chunks.jsonl.

This is the baseline every later retrieval number gets compared against, and the lexical leg
of hybrid retrieval later. No model download, so it cold-starts fast on scale-to-zero hosting.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ukmoney_rag.models import Chunk

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens; '£12,570' -> '12570' so amounts match however they're typed."""
    text = _THOUSANDS.sub("", text.lower())
    return [t for t in _TOKEN.findall(text) if t not in _STOPWORDS]


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._chunks = list(chunks)
        self._k1, self._b = k1, b
        self._tfs = [Counter(tokenize(c.embed_text)) for c in self._chunks]
        self._lengths = [sum(tf.values()) for tf in self._tfs]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 1.0
        doc_freq: Counter[str] = Counter()
        for tf in self._tfs:
            doc_freq.update(tf.keys())
        n = len(self._chunks)
        self._idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def __len__(self) -> int:
        return len(self._chunks)

    @classmethod
    def from_jsonl(cls, path: Path) -> BM25Index:
        lines = path.read_text(encoding="utf-8").splitlines()
        return cls([Chunk.model_validate_json(line) for line in lines if line.strip()])

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        terms = list(dict.fromkeys(tokenize(query)))  # dedupe, keep order
        if not terms:
            return []
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tfs):
            norm = self._k1 * (1 - self._b + self._b * self._lengths[i] / self._avg_len)
            score = sum(
                self._idf[t] * tf[t] * (self._k1 + 1) / (tf[t] + norm) for t in terms if t in tf
            )
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))  # deterministic tie-break
        return [SearchHit(self._chunks[i], round(s, 4)) for s, i in scored[:k]]

"""Short blind questions must be compressions of the long ones, not rewrites.

Each line of eval/blind_questions_short.md is the matching line of
eval/blind_questions.md shortened by deleting words. This guard stops corpus
vocabulary (official benefit and tax names) leaking into the blind set.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
LONG = EVAL_DIR / "blind_questions.md"
SHORT = EVAL_DIR / "blind_questions_short.md"
N_QUESTIONS = 30
MAX_WORDS = 15

# Glue words allowed when restructuring a sentence. No content words.
ALLOWED_NEW = frozenset(
    {
        "a", "an", "the", "i", "my", "me", "is", "it", "do", "does", "can", "will",
        "be", "to", "for", "of", "on", "in", "if", "and", "or", "what", "how",
        "much", "any", "get", "still",
    }
)  # fmt: skip
WORD = re.compile(r"[a-z0-9£']+")


def _questions(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def _words(text: str) -> list[str]:
    return WORD.findall(text.lower().replace("\u2019", "'"))


def test_files_are_aligned() -> None:
    assert len(_questions(LONG)) == N_QUESTIONS
    assert len(_questions(SHORT)) == N_QUESTIONS


@pytest.mark.parametrize("i", range(N_QUESTIONS))
def test_short_is_compression_of_long(i: int) -> None:
    long_q, short_q = _questions(LONG)[i], _questions(SHORT)[i]
    words = _words(short_q)
    assert len(words) <= MAX_WORDS, f"question {i + 1}: {len(words)} words (max {MAX_WORDS})"
    new = set(words) - set(_words(long_q)) - ALLOWED_NEW
    assert not new, f"question {i + 1}: words not in the long version: {sorted(new)}"

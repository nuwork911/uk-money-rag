"""Blind cases in eval/cases.toml must quote their question exactly, by number.

Case ids are blind-NN-<slug> (NN = line order in eval/blind_questions.md) and
blind-NN-<slug>-short. A copy-paste slip while labelling would silently reword a frozen
question; this catches it. Flip LABELLING_COMPLETE once all 30 pairs are labelled.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

EVAL = Path(__file__).resolve().parents[1] / "eval"
N_QUESTIONS = 30
LABELLING_COMPLETE = False
BLIND_ID = re.compile(r"^blind-(\d{2})-[a-z0-9-]+?(-short)?$")


def _questions(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def _blind_cases() -> list[dict[str, Any]]:
    data = tomllib.loads((EVAL / "cases.toml").read_text(encoding="utf-8"))
    return [c for c in data.get("case", []) if str(c["id"]).startswith("blind-")]


def test_blind_cases_quote_their_question_exactly() -> None:
    long_qs = _questions(EVAL / "blind_questions.md")
    short_qs = _questions(EVAL / "blind_questions_short.md")
    seen: set[tuple[int, bool]] = set()
    for case in _blind_cases():
        m = BLIND_ID.match(case["id"])
        assert m, f"{case['id']}: expected blind-NN-<slug> or blind-NN-<slug>-short"
        n, is_short = int(m.group(1)), m.group(2) is not None
        assert 1 <= n <= N_QUESTIONS, f"{case['id']}: question number out of range"
        assert (n, is_short) not in seen, f"{case['id']}: question {n} labelled twice"
        seen.add((n, is_short))
        expected = (short_qs if is_short else long_qs)[n - 1]
        assert case["question"] == expected, f"{case['id']}: question text differs from file"
        origin = "short" if is_short else "blind"
        assert origin in case["tags"], f"{case['id']}: missing tag {origin!r}"
    if LABELLING_COMPLETE:
        missing = {(n, s) for n in range(1, N_QUESTIONS + 1) for s in (False, True)} - seen
        assert not missing, f"unlabelled: {sorted(missing)}"

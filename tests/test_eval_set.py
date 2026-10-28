"""Guards the real eval set: every label must resolve against the pinned snapshot.

Offline and fast (no model), so it runs in CI on every PR. If a snapshot refresh moves or
rewords an answer, this fails and the label gets fixed, instead of the case silently scoring 0.
"""

from __future__ import annotations

from pathlib import Path

from ukmoney_rag.evaluation import main

ROOT = Path(__file__).resolve().parents[1]


def test_eval_set_resolves_against_pinned_snapshot() -> None:
    assert (
        main(
            [
                "--data-dir",
                str(ROOT / "data/snapshot"),
                "--cases",
                str(ROOT / "eval/cases.toml"),
                "--pin",
                str(ROOT / "eval/CORPUS_SHA256"),
                "validate",
                "--known-failures",
                str(ROOT / "eval/known_failures.toml"),
            ]
        )
        == 0
    )

"""Print a TOML skeleton for one blind question: the long case and its -short variant.

The question text is copied from eval/blind_questions*.md by number, never retyped, so
labelling cannot reword a question. Append the output to eval/cases.toml and fill in the
TODOs. tests/test_blind_cases.py checks the text still matches.

    .venv/bin/python scripts/blind_case.py 25 tax-free-savings-yearly >> eval/cases.toml
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1] / "eval"
HEADER = "# --- blind cases (eval/README.md, 'Blind set') "
SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def questions(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def toml_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)  # JSON string escapes are valid TOML


def skeleton(n: int, slug: str) -> str:
    long_q = questions(EVAL / "blind_questions.md")[n - 1]
    short_q = questions(EVAL / "blind_questions_short.md")[n - 1]
    cid = f"blind-{n:02d}-{slug}"
    out = []
    if HEADER not in (EVAL / "cases.toml").read_text(encoding="utf-8"):
        out.append(f"\n{HEADER}{'-' * (99 - len(HEADER))}\n")
    out.append(
        f"""
[[case]]
id = "{cid}"
question = {toml_str(long_q)}
tags = ["TODO-topic", "TODO-lexical-or-paraphrase", "blind"]
notes = "TODO or delete this line"
[[case.evidence]]
url = "TODO"
quote = "TODO"

[[case]]
id = "{cid}-short"
question = {toml_str(short_q)}
tags = ["short"]
notes = "Short variant of {cid}; evidence = first ask of the long case."
[[case.evidence]]
url = "TODO"
quote = "TODO"
"""
    )
    return "".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("n", type=int, help="question number, 1-30 (order in blind_questions.md)")
    p.add_argument("slug", help="short lowercase-hyphen name, e.g. emergency-cash-eviction")
    a = p.parse_args()
    total = len(questions(EVAL / "blind_questions.md"))
    if not 1 <= a.n <= total:
        p.error(f"n must be 1-{total}")
    if not SLUG.match(a.slug):
        p.error("slug must be lowercase letters/digits separated by hyphens")
    print(skeleton(a.n, a.slug), end="")


if __name__ == "__main__":
    main()

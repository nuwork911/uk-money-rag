from __future__ import annotations

import json
from pathlib import Path

import pytest

from fakes import FakeEmbedder, make_chunk
from ukmoney_rag import evaluation as ev
from ukmoney_rag.api_dense import DenseRetriever
from ukmoney_rag.dense import DenseIndex, build_index, load_chunks
from ukmoney_rag.evaluation import (
    BM25Retriever,
    DenseEvalRetriever,
    EvalCase,
    EvalSet,
    Evidence,
    ResolveError,
    compare,
    normalize,
    render_summary,
    resolve,
    run_eval,
)

CHUNKS = [
    make_chunk(0, "Child Benefit", "You can get £26.05 a week for your eldest child."),
    make_chunk(1, "State Pension", "The full rate of new State Pension is £241.30 a week."),
    make_chunk(2, "Pension Credit", "Tops up your weekly income if you\u2019re on a low income."),
    make_chunk(
        3, "Council Tax", "Council Tax is paid to your local council in 10 monthly instalments."
    ),
]

CASES_TOML = """
[[case]]
id = "pension-rate"
question = "full new state pension weekly rate"
tags = ["pensions"]
[[case.evidence]]
url = "https://www.gov.uk/doc1"
quote = "full rate of new State Pension"

[[case]]
id = "child-benefit"
question = "child benefit eldest child amount"
tags = ["benefits"]
[[case.evidence]]
url = "https://www.gov.uk/doc0"
quote = "a week for your eldest child"

[[case]]
id = "out-of-corpus"
question = "how do I appeal a parking fine"
answerable = false
"""


@pytest.fixture
def eval_set(tmp_path: Path) -> EvalSet:
    path = tmp_path / "cases.toml"
    path.write_text(CASES_TOML, encoding="utf-8")
    return EvalSet.load(path)


def test_normalize_is_forgiving_about_typography() -> None:
    assert normalize("If  you\u2019re\non a LOW income") == normalize("if you're on a low income")


def test_resolve_matches_url_and_quote(eval_set: EvalSet) -> None:
    gold = resolve(eval_set.answerable, CHUNKS)
    assert gold == {"pension-rate": {CHUNKS[1].chunk_id}, "child-benefit": {CHUNKS[0].chunk_id}}


def test_resolve_reports_every_miss_including_wrong_url() -> None:
    cases = [
        EvalCase(
            id="a", question="q", evidence=(Evidence(url="https://x", quote="not in any chunk"),)
        ),
        EvalCase(  # right quote, wrong page: still a miss
            id="b",
            question="q",
            evidence=(
                Evidence(url="https://www.gov.uk/doc0", quote="full rate of new State Pension"),
            ),
        ),
    ]
    with pytest.raises(ResolveError) as exc:
        resolve(cases, CHUNKS)
    assert "a: no chunk" in str(exc.value)
    assert "b: no chunk" in str(exc.value)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ({"id": "Bad_Id", "question": "q", "answerable": False}, "slug"),
        ({"id": "a", "question": "q"}, "needs at least one evidence"),
        (
            {
                "id": "a",
                "question": "q",
                "answerable": False,
                "evidence": [{"url": "u", "quote": "a b c"}],
            },
            "must not have evidence",
        ),
        (
            {"id": "a", "question": "q", "evidence": [{"url": "u", "quote": "too short"}]},
            "3-30 words",
        ),
        ({"id": "a", "question": "q", "answerable": False, "typo_field": 1}, "Extra inputs"),
    ],
)
def test_case_schema_rejects_bad_labels(case: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EvalCase.model_validate(case)


def test_eval_set_rejects_duplicates_and_empty(tmp_path: Path) -> None:
    dup = tmp_path / "dup.toml"
    first_case = CASES_TOML.split("[[case]]")[1]
    dup.write_text(CASES_TOML + "[[case]]" + first_case, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case ids"):
        EvalSet.load(dup)
    empty = tmp_path / "empty.toml"
    empty.write_text("# nothing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no \\[\\[case\\]\\]"):
        EvalSet.load(empty)


def test_run_eval_bm25_scores_and_records_provenance(eval_set: EvalSet) -> None:
    run = run_eval(eval_set, CHUNKS, "c" * 64, BM25Retriever(CHUNKS), git_sha="abc1234")
    assert run.overall.n == 2
    assert run.skipped_unanswerable == ("out-of-corpus",)
    assert run.overall.mrr == 1.0  # both questions share distinctive words with their chunk
    assert run.overall.hit_rate[1] == 1.0
    assert run.by_tag["pensions"].n == 1
    assert run.eval_set_sha256 == eval_set.sha256
    assert run.retriever_config == {"k1": "1.5", "b": "0.75"}
    assert run.cases[0].retrieved[0] == CHUNKS[1].chunk_id
    assert run.overall.mean_words[1] == pytest.approx(
        (CHUNKS[1].n_words + CHUNKS[0].n_words) / 2, abs=0.1
    )


def test_dense_adapter_runs_through_serving_retriever(eval_set: EvalSet, tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text("".join(c.model_dump_json() + "\n" for c in CHUNKS), encoding="utf-8")
    chunks, sha = load_chunks(path)
    index_dir = build_index(chunks, sha, FakeEmbedder(), tmp_path / "indexes")
    dense = DenseEvalRetriever(DenseRetriever(DenseIndex.load(index_dir, sha), FakeEmbedder()))
    run = run_eval(eval_set, chunks, sha, dense)
    assert run.retriever == "dense"
    assert run.retriever_config["index_id"] == index_dir.name
    assert run.overall.n == 2


def test_compare_and_summary(eval_set: EvalSet) -> None:
    a = run_eval(eval_set, CHUNKS, "c" * 64, BM25Retriever(CHUNKS))
    b = run_eval(eval_set, CHUNKS, "c" * 64, BM25Retriever(CHUNKS, k1=0.9))
    cis = compare(a, b)
    assert cis["mrr"].mean_diff == 0.0
    assert not cis["hit@5"].excludes_zero
    md = render_summary([a, b])
    assert "| bm25 | k1=1.5, b=0.75 | 1.000 |" in md
    assert "no detectable difference" in md
    assert "| pension-rate | 1 | 1 |" in md


def test_compare_refuses_different_eval_sets(eval_set: EvalSet) -> None:
    a = run_eval(eval_set, CHUNKS, "c" * 64, BM25Retriever(CHUNKS))
    b = a.model_copy(update={"eval_set_sha256": "other"})
    with pytest.raises(ValueError, match="not comparable"):
        compare(a, b)
    c = a.model_copy(update={"cases": a.cases[:1]})
    with pytest.raises(ValueError, match="different cases"):
        compare(a, c)


# --- CLI, end to end on the ingest fixture corpus -------------------------------------------


@pytest.fixture
def cli_env(processed_dir: Path, tmp_path: Path) -> list[str]:
    chunks, sha = load_chunks(processed_dir / "chunks.jsonl")
    target = chunks[0]
    quote = " ".join(target.text.split()[:6])
    (tmp_path / "pin").write_text(sha + "\n", encoding="utf-8")
    (tmp_path / "cases.toml").write_text(
        f'[[case]]\nid = "fixture"\nquestion = "{quote}"\n'
        f'[[case.evidence]]\nurl = "{target.url}"\nquote = "{quote}"\n',
        encoding="utf-8",
    )
    (tmp_path / "kf.toml").write_text(
        '[[failure]]\nid = "x"\nfound = "f"\nquery = "q"\nfailure = "f"\nexpected = "e"\n'
        'hypothesis = "h"\nstatus = "open"\neval_cases = ["fixture"]\n',
        encoding="utf-8",
    )
    return [
        "--data-dir",
        str(processed_dir),
        "--cases",
        str(tmp_path / "cases.toml"),
        "--pin",
        str(tmp_path / "pin"),
    ]


def test_cli_validate_find_run(
    cli_env: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ev.main([*cli_env, "validate", "--known-failures", str(tmp_path / "kf.toml")]) == 0
    assert "eval set OK: 1 cases" in capsys.readouterr().out

    assert ev.main([*cli_env, "find", "zzz not present zzz"]) == 1
    quote = (tmp_path / "cases.toml").read_text().split('quote = "')[1].split('"')[0]
    assert ev.main([*cli_env, "find", quote]) == 0
    assert "1 chunk(s)" in capsys.readouterr().out

    out = tmp_path / "results"
    assert ev.main([*cli_env, "run", "--retrievers", "bm25", "--out", str(out)]) == 0
    result = json.loads((out / "bm25.json").read_text())
    assert result["overall"]["n"] == 1
    assert (out / "summary.md").read_text().startswith("# Retrieval eval results")


def test_cli_validate_flags_dangling_known_failures(cli_env: list[str], tmp_path: Path) -> None:
    kf = tmp_path / "kf.toml"
    kf.write_text(kf.read_text().replace('["fixture"]', '["nope"]'), encoding="utf-8")
    assert ev.main([*cli_env, "validate", "--known-failures", str(kf)]) == 1
    kf.write_text(kf.read_text().replace('eval_cases = ["nope"]\n', ""), encoding="utf-8")
    assert ev.main([*cli_env, "validate", "--known-failures", str(kf)]) == 1


def test_cli_refuses_unpinned_corpus_and_unknown_retriever(
    cli_env: list[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="unknown retriever"):
        ev.main([*cli_env, "run", "--retrievers", "bm25,splade", "--out", str(tmp_path / "o")])
    (tmp_path / "pin").write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="labels may be stale"):
        ev.main([*cli_env, "validate", "--known-failures", str(tmp_path / "kf.toml")])


def test_cli_returns_1_on_unresolvable_label(cli_env: list[str], tmp_path: Path) -> None:
    cases = tmp_path / "cases.toml"
    cases.write_text(
        cases.read_text().replace('url = "', 'url = "https://wrong/'), encoding="utf-8"
    )
    assert ev.main([*cli_env, "validate", "--known-failures", str(tmp_path / "kf.toml")]) == 1


def test_git_state_falls_back_outside_a_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_SHA", "deadbee")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert ev.git_state() == ("deadbee", False)

"""Retrieval evaluation: hand-written cases -> gold chunks -> per-retriever metrics.

Gold labels are *evidence quotes*, not chunk IDs. Each case lists one or more alternative
places its answer appears (GOV.UK page URL + a short verbatim quote). A chunk is relevant if it
comes from one of those URLs and contains that quote; any one relevant chunk answers the case.

Why not chunk IDs: they are content hashes, so every one changes when the chunker changes.
Quotes survive re-chunking, which is what makes the chunk-size ablation a fair comparison.
``validate`` resolves every quote against the pinned snapshot, so a label that stops matching
fails CI instead of silently scoring zero.

CLI (``ukmoney-eval``):
    validate              check cases + known failures against data/snapshot (CI runs this)
    find QUOTE [--url U]  labelling helper: which chunks contain this quote?
    run                   score retrievers, write eval/results/{<name>.json, summary.md}
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tomllib
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ukmoney_rag.api_dense import DenseRetriever
from ukmoney_rag.dense import DenseIndex, load_chunks, resolve_index_dir
from ukmoney_rag.metrics import (
    BootstrapCI,
    first_relevant_rank,
    hit_at_k,
    paired_bootstrap,
    reciprocal_rank,
)
from ukmoney_rag.models import Chunk
from ukmoney_rag.search import BM25Index

KS: tuple[int, ...] = (1, 3, 5, 10)
K_MAX = max(KS)
_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_QUOTE_WORDS = (3, 30)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Evidence(_Frozen):
    url: str
    quote: str

    @field_validator("quote")
    @classmethod
    def _short_quote(cls, v: str) -> str:
        n = len(v.split())
        lo, hi = _QUOTE_WORDS
        if not lo <= n <= hi:
            raise ValueError(
                f"quote must be {lo}-{hi} words (got {n}); quote the answer, not a page"
            )
        return v


class EvalCase(_Frozen):
    id: str
    question: str
    tags: tuple[str, ...] = ()
    answerable: bool = True
    evidence: tuple[Evidence, ...] = ()
    notes: str | None = None

    @field_validator("id")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _ID.match(v):
            raise ValueError(f"case id {v!r} must be a lowercase-hyphen slug")
        return v

    @model_validator(mode="after")
    def _evidence_iff_answerable(self) -> EvalCase:
        if self.answerable and not self.evidence:
            raise ValueError(f"{self.id}: answerable case needs at least one evidence item")
        if not self.answerable and self.evidence:
            raise ValueError(f"{self.id}: unanswerable case must not have evidence")
        return self


class KnownFailure(_Frozen):
    id: str
    found: str
    query: str
    failure: str
    expected: str
    hypothesis: str
    status: str = Field(pattern=r"^(open|fixed|regression-test)$")
    eval_cases: tuple[str, ...] = ()
    test: str | None = None  # pytest node id, for failures guarded by a unit test instead


class EvalSet(BaseModel):
    cases: tuple[EvalCase, ...]
    sha256: str  # of the cases file bytes: results are only comparable for the same eval set

    @classmethod
    def load(cls, path: Path) -> EvalSet:
        raw = path.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
        cases = tuple(EvalCase.model_validate(c) for c in data.get("case", []))
        if not cases:
            raise ValueError(f"{path} has no [[case]] entries")
        ids = [c.id for c in cases]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate case ids: {dupes}")
        return cls(cases=cases, sha256=hashlib.sha256(raw).hexdigest())

    @property
    def answerable(self) -> tuple[EvalCase, ...]:
        return tuple(c for c in self.cases if c.answerable)


def load_known_failures(path: Path) -> tuple[KnownFailure, ...]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return tuple(KnownFailure.model_validate(kf) for kf in data.get("failure", []))


# --- resolving labels -------------------------------------------------------------------------

_PUNCT = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-"})


def normalize(text: str) -> str:
    """Forgiving match: NFKC, straight quotes, collapsed whitespace, casefolded."""
    text = unicodedata.normalize("NFKC", text).translate(_PUNCT)
    return " ".join(text.split()).casefold()


def resolve_evidence(ev: Evidence, chunks: Iterable[Chunk]) -> frozenset[str]:
    needle = normalize(ev.quote)
    return frozenset(c.chunk_id for c in chunks if c.url == ev.url and needle in normalize(c.text))


class ResolveError(ValueError):
    """One or more evidence quotes match no chunk in the corpus."""


def resolve(cases: Sequence[EvalCase], chunks: Sequence[Chunk]) -> dict[str, frozenset[str]]:
    """case id -> relevant chunk IDs. Every evidence item must match; reports all misses at once."""
    out: dict[str, frozenset[str]] = {}
    misses: list[str] = []
    for case in cases:
        groups = [resolve_evidence(ev, chunks) for ev in case.evidence]
        misses += [
            f"{case.id}: no chunk at {ev.url} contains {ev.quote!r}"
            for ev, g in zip(case.evidence, groups, strict=True)
            if not g
        ]
        out[case.id] = frozenset().union(*groups)
    if misses:
        raise ResolveError("unresolved evidence:\n  " + "\n  ".join(misses))
    return out


# --- retrievers -------------------------------------------------------------------------------


class Retriever(Protocol):
    @property
    def name(self) -> str: ...

    def config(self) -> dict[str, str]: ...

    def search(self, query: str, k: int) -> list[str]: ...


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self._index = BM25Index(chunks, k1=k1, b=b)
        self._k1, self._b = k1, b

    def config(self) -> dict[str, str]:
        return {"k1": str(self._k1), "b": str(self._b)}

    def search(self, query: str, k: int) -> list[str]:
        return [hit.chunk.chunk_id for hit in self._index.search(query, k)]


class DenseEvalRetriever:
    """Adapter over the serving DenseRetriever, so eval scores exactly what production runs."""

    name = "dense"

    def __init__(self, retriever: DenseRetriever) -> None:
        self._r = retriever

    def config(self) -> dict[str, str]:
        m = self._r.index.manifest
        return {"index_id": m.index_id, "model_id": m.model_id, "model_sha256": m.model_sha256}

    def search(self, query: str, k: int) -> list[str]:
        return [chunk_id for chunk_id, _ in self._r.search(query, k)]


def _real_dense(  # pragma: no cover
    index_root: Path, model_cache: Path, corpus_sha256: str
) -> Retriever:
    """Loads the real ONNX model; exercised by `make eval`, not by unit tests."""
    from ukmoney_rag.embed import FastEmbedEmbedder

    index = DenseIndex.load(resolve_index_dir(index_root), corpus_sha256)
    embedder = FastEmbedEmbedder(cache_dir=model_cache)
    return DenseEvalRetriever(DenseRetriever(index, embedder))


# --- running ----------------------------------------------------------------------------------


class CaseResult(_Frozen):
    case_id: str
    tags: tuple[str, ...]
    first_relevant_rank: int | None  # None = not in top K_MAX
    reciprocal_rank: float
    hit: dict[int, float]
    retrieved: tuple[str, ...]


class Aggregate(_Frozen):
    n: int
    mrr: float
    hit_rate: dict[int, float]
    mean_words: dict[int, float]  # words returned in the top k: bigger chunks "win" hits cheaply


class RunResult(_Frozen):
    retriever: str
    retriever_config: dict[str, str]
    eval_set_sha256: str
    corpus_sha256: str
    n_chunks: int
    git_sha: str
    git_dirty: bool
    k_max: int
    skipped_unanswerable: tuple[str, ...]
    overall: Aggregate
    by_tag: dict[str, Aggregate]
    cases: tuple[CaseResult, ...]


def _r4(x: float) -> float:
    return round(x, 4)


def _aggregate(results: Sequence[CaseResult], words: Mapping[str, int]) -> Aggregate:
    n = len(results)
    return Aggregate(
        n=n,
        mrr=_r4(sum(r.reciprocal_rank for r in results) / n),
        hit_rate={k: _r4(sum(r.hit[k] for r in results) / n) for k in KS},
        mean_words={
            k: round(sum(sum(words[c] for c in r.retrieved[:k]) for r in results) / n, 1)
            for k in KS
        },
    )


def run_eval(
    eval_set: EvalSet,
    chunks: Sequence[Chunk],
    corpus_sha256: str,
    retriever: Retriever,
    *,
    git_sha: str = "unknown",
    git_dirty: bool = False,
) -> RunResult:
    gold = resolve(eval_set.answerable, chunks)
    words = {c.chunk_id: c.n_words for c in chunks}
    results: list[CaseResult] = []
    for case in eval_set.answerable:
        ranked = retriever.search(case.question, K_MAX)
        results.append(
            CaseResult(
                case_id=case.id,
                tags=case.tags,
                first_relevant_rank=first_relevant_rank(ranked, gold[case.id]),
                reciprocal_rank=_r4(reciprocal_rank(ranked, gold[case.id], K_MAX)),
                hit={k: hit_at_k(ranked, gold[case.id], k) for k in KS},
                retrieved=tuple(ranked),
            )
        )
    tags = sorted({t for r in results for t in r.tags})
    return RunResult(
        retriever=retriever.name,
        retriever_config=retriever.config(),
        eval_set_sha256=eval_set.sha256,
        corpus_sha256=corpus_sha256,
        n_chunks=len(chunks),
        git_sha=git_sha,
        git_dirty=git_dirty,
        k_max=K_MAX,
        skipped_unanswerable=tuple(c.id for c in eval_set.cases if not c.answerable),
        overall=_aggregate(results, words),
        by_tag={t: _aggregate([r for r in results if t in r.tags], words) for t in tags},
        cases=tuple(results),
    )


def compare(base: RunResult, other: RunResult, seed: int = 0) -> dict[str, BootstrapCI]:
    """Paired bootstrap of other - base on MRR and hit@5, over identical cases."""
    if (base.eval_set_sha256, base.corpus_sha256) != (other.eval_set_sha256, other.corpus_sha256):
        raise ValueError("runs used different eval sets or corpora; not comparable")
    if [c.case_id for c in base.cases] != [c.case_id for c in other.cases]:
        raise ValueError("runs cover different cases")
    return {
        "mrr": paired_bootstrap(
            [c.reciprocal_rank for c in base.cases],
            [c.reciprocal_rank for c in other.cases],
            seed=seed,
        ),
        "hit@5": paired_bootstrap(
            [c.hit[5] for c in base.cases], [c.hit[5] for c in other.cases], seed=seed
        ),
    }


# --- reporting --------------------------------------------------------------------------------


def _rank(r: int | None) -> str:
    return "-" if r is None else str(r)


def render_summary(runs: Sequence[RunResult]) -> str:
    """Markdown summary. The first run is the baseline for the bootstrap comparison."""
    base = runs[0]
    lines = [
        "# Retrieval eval results",
        "",
        "Generated by `make eval`; do not edit by hand.",
        "",
        f"- corpus `{base.corpus_sha256[:12]}` ({base.n_chunks} chunks)",
        f"- eval set `{base.eval_set_sha256[:12]}`: {base.overall.n} answerable cases"
        f" (+{len(base.skipped_unanswerable)} unanswerable, not scored)",
        f"- code `{base.git_sha}`"
        + (" (dirty tree: not a citable result)" if base.git_dirty else ""),
        "",
        "| retriever | config | MRR@10 | hit@1 | hit@3 | hit@5 | hit@10 | words in top 5 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        cfg = ", ".join(
            f"{k}={v[:12] if k.endswith(('sha256', '_id')) and '/' not in v else v}"
            for k, v in r.retriever_config.items()
        )
        rec = " | ".join(f"{r.overall.hit_rate[k]:.3f}" for k in KS)
        words = r.overall.mean_words[5]
        lines.append(f"| {r.retriever} | {cfg} | {r.overall.mrr:.3f} | {rec} | {words:.0f} |")
    for other in runs[1:]:
        lines += ["", f"## {other.retriever} vs {base.retriever} (paired bootstrap, 95% CI)", ""]
        for metric, ci in compare(base, other).items():
            verdict = "difference" if ci.excludes_zero else "no detectable difference"
            lines.append(
                f"- {metric}: {ci.mean_diff:+.3f} [{ci.low:+.3f}, {ci.high:+.3f}]"
                f" over {ci.n_cases} cases ({ci.n_boot} resamples, seed {ci.seed}): {verdict}"
            )
    tags = sorted({t for r in runs for t in r.by_tag})
    if tags:
        lines += [
            "",
            "## MRR@10 by tag",
            "",
            "| tag | n | " + " | ".join(r.retriever for r in runs) + " |",
        ]
        lines.append("|---|---|" + "---|" * len(runs))
        for t in tags:
            n = base.by_tag[t].n if t in base.by_tag else 0
            vals = " | ".join(f"{r.by_tag[t].mrr:.3f}" if t in r.by_tag else "-" for r in runs)
            lines.append(f"| {t} | {n} | {vals} |")
    lines += [
        "",
        "## Rank of first relevant chunk per case (- = not in top 10)",
        "",
        "| case | " + " | ".join(r.retriever for r in runs) + " |",
        "|---|" + "---|" * len(runs),
    ]
    for i, case in enumerate(base.cases):
        lines.append(
            f"| {case.case_id} | "
            + " | ".join(_rank(r.cases[i].first_relevant_rank) for r in runs)
            + " |"
        )
    return "\n".join(lines) + "\n"


# --- CLI --------------------------------------------------------------------------------------


def git_state() -> tuple[str, bool]:
    """(short sha, dirty?) ignoring untracked files and eval/results itself."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!eval/results"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GIT_SHA", "unknown"), False
    return sha, bool(status.strip())


def _load_pinned(data_dir: Path, pin: Path) -> tuple[list[Chunk], str]:
    chunks, sha = load_chunks(data_dir / "chunks.jsonl")
    expected = pin.read_text(encoding="utf-8").strip()
    if sha != expected:
        raise SystemExit(
            f"corpus {sha[:12]} != pinned {expected[:12]} ({pin}); labels may be stale"
        )
    return chunks, sha


def _validate(args: argparse.Namespace) -> int:
    chunks, _ = _load_pinned(args.data_dir, args.pin)
    eval_set = EvalSet.load(args.cases)
    gold = resolve(eval_set.answerable, chunks)
    ids = {c.id for c in eval_set.cases}
    problems = []
    for kf in load_known_failures(args.known_failures):
        missing = [c for c in kf.eval_cases if c not in ids]
        if missing:
            problems.append(f"known failure {kf.id} points at missing cases {missing}")
        if kf.status == "open" and not kf.eval_cases:
            problems.append(f"known failure {kf.id} is open but has no eval case")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    n_gold = sum(len(g) for g in gold.values())
    print(
        f"eval set OK: {len(eval_set.cases)} cases ({len(eval_set.answerable)} answerable), "
        f"{n_gold} gold chunk matches, sha256={eval_set.sha256[:12]}"
    )
    return 0


def _find(args: argparse.Namespace) -> int:
    chunks, _ = load_chunks(args.data_dir / "chunks.jsonl")
    needle = normalize(args.quote)
    hits = [
        c for c in chunks if needle in normalize(c.text) and (args.url is None or c.url == args.url)
    ]
    for c in hits:
        text = " ".join(c.text.split())
        at = normalize(text).find(needle)
        print(
            f"{c.chunk_id}  {c.n_words:>3}w  {c.url}\n    ...{text[max(0, at - 80) : at + 120]}..."
        )
    print(f"{len(hits)} chunk(s)")
    return 0 if hits else 1


def _run(args: argparse.Namespace) -> int:
    chunks, sha = _load_pinned(args.data_dir, args.pin)
    eval_set = EvalSet.load(args.cases)
    sha_git, dirty = git_state()
    retrievers: list[Retriever] = []
    for name in args.retrievers.split(","):
        if name == "bm25":
            retrievers.append(BM25Retriever(chunks))
        elif name == "dense":
            retrievers.append(_real_dense(args.index_root, args.model_cache, sha))
        else:
            raise SystemExit(f"unknown retriever {name!r} (choose from bm25,dense)")
    runs = [
        run_eval(eval_set, chunks, sha, r, git_sha=sha_git, git_dirty=dirty) for r in retrievers
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    for run in runs:
        (args.out / f"{run.retriever}.json").write_text(
            run.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    summary = render_summary(runs)
    (args.out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    if dirty:
        print("WARNING: uncommitted changes; commit first for a citable result.", file=sys.stderr)
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ukmoney-eval", description="Retrieval evaluation")
    p.add_argument("--data-dir", type=Path, default=Path("data/snapshot"))
    p.add_argument("--cases", type=Path, default=Path("eval/cases.toml"))
    p.add_argument("--pin", type=Path, default=Path("eval/CORPUS_SHA256"))
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="resolve every label against the pinned snapshot")
    v.add_argument("--known-failures", type=Path, default=Path("eval/known_failures.toml"))
    f = sub.add_parser("find", help="labelling helper: chunks containing a quote")
    f.add_argument("quote")
    f.add_argument("--url", default=None, help="restrict to one page URL")
    r = sub.add_parser("run", help="score retrievers and write results")
    r.add_argument("--retrievers", default="bm25,dense", help="comma-separated; first = baseline")
    r.add_argument("--out", type=Path, default=Path("eval/results"))
    r.add_argument("--index-root", type=Path, default=Path("indexes"))
    r.add_argument("--model-cache", type=Path, default=Path(".cache/models"))
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {"validate": _validate, "find": _find, "run": _run}
    try:
        return handlers[args.cmd](args)
    except (ResolveError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

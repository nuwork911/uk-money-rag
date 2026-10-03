"""CLI: fetch -> parse -> chunk -> data/processed/{chunks.jsonl, manifest.json}.

The manifest records what produced the output (config, per-source hashes, output hash) so any
result can be traced back to exact inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import statistics
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from ukmoney_rag.chunk import ChunkConfig, chunk_document
from ukmoney_rag.fetch import USER_AGENT, Fetcher, FetchError
from ukmoney_rag.models import Chunk, Source
from ukmoney_rag.parse import ParseError, parse_document

logger = logging.getLogger("ukmoney_rag.ingest")


class SourceReport(BaseModel):
    name: str
    url: str
    licence: str
    fetched_at: datetime
    from_cache: bool
    raw_sha256: str
    n_sections: int
    n_chunks: int


class Manifest(BaseModel):
    created_at: datetime
    chunk_config: dict[str, int]
    n_documents: int
    n_chunks: int
    words_min: int
    words_median: float
    words_max: int
    sources: list[SourceReport]
    failures: dict[str, str]
    chunks_sha256: str


@dataclass
class IngestResult:
    chunks: list[Chunk] = field(default_factory=list)
    reports: list[SourceReport] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    required_failures: dict[str, str] = field(default_factory=dict)


def load_sources(path: Path) -> list[Source]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    sources = [Source.model_validate(entry) for entry in data.get("source", [])]
    names = [s.name for s in sources]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate source names in {path}")
    return sources


def run_ingest(sources: Sequence[Source], fetcher: Fetcher, cfg: ChunkConfig) -> IngestResult:
    result = IngestResult()
    for source in sources:
        try:
            fetched = fetcher.fetch(source)
            doc = parse_document(source, fetched.content, fetched.fetched_at)
        except (FetchError, ParseError) as exc:
            level = logging.WARNING if source.optional else logging.ERROR
            logger.log(level, "%s failed: %s", source.name, exc)
            result.failures[source.name] = str(exc)
            if not source.optional:
                result.required_failures[source.name] = str(exc)
            continue
        chunks = chunk_document(doc, cfg)
        logger.info("%s: %d sections -> %d chunks", source.name, len(doc.sections), len(chunks))
        result.chunks.extend(chunks)
        result.reports.append(
            SourceReport(
                name=source.name,
                url=source.url,
                licence=source.licence,
                fetched_at=fetched.fetched_at,
                from_cache=fetched.from_cache,
                raw_sha256=hashlib.sha256(fetched.content.encode()).hexdigest(),
                n_sections=len(doc.sections),
                n_chunks=len(chunks),
            )
        )
    return result


def write_outputs(result: IngestResult, cfg: ChunkConfig, out_dir: Path) -> Manifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for chunk in result.chunks:
            fh.write(chunk.model_dump_json() + "\n")

    words = [c.n_words for c in result.chunks] or [0]
    manifest = Manifest(
        created_at=datetime.now(UTC),
        chunk_config=asdict(cfg),
        n_documents=len(result.reports),
        n_chunks=len(result.chunks),
        words_min=min(words),
        words_median=statistics.median(words),
        words_max=max(words),
        sources=result.reports,
        failures=result.failures,
        chunks_sha256=hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ukmoney-ingest", description=__doc__)
    p.add_argument("--sources", type=Path, default=Path("configs/sources.toml"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--offline", action="store_true", help="use cache only; never touch the network")
    p.add_argument("--refresh", action="store_true", help="ignore cache and re-download")
    p.add_argument("--max-words", type=int, default=ChunkConfig.max_words)
    p.add_argument("--overlap-words", type=int, default=ChunkConfig.overlap_words)
    p.add_argument("--user-agent", default=USER_AGENT)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = ChunkConfig(max_words=args.max_words, overlap_words=args.overlap_words)
    sources = load_sources(args.sources)
    fetcher = Fetcher(
        args.cache_dir, offline=args.offline, refresh=args.refresh, user_agent=args.user_agent
    )
    try:
        result = run_ingest(sources, fetcher, cfg)
    finally:
        fetcher.close()

    manifest = write_outputs(result, cfg, args.out_dir)
    logger.info(
        "wrote %d chunks from %d documents (words/chunk min=%d median=%.0f max=%d) -> %s",
        manifest.n_chunks,
        manifest.n_documents,
        manifest.words_min,
        manifest.words_median,
        manifest.words_max,
        args.out_dir,
    )
    if result.required_failures:
        logger.error("required sources failed: %s", ", ".join(result.required_failures))
        return 1
    if not result.chunks:
        logger.error("no chunks produced")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

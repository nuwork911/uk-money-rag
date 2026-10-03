# uk-money-rag

Grounded question answering over UK public money guidance (GOV.UK), built as a
production-style pipeline: typed, tested, and reproducible.

> **Status: Week 1 of 5.** Ingestion plus a keyword-search (BM25) baseline, with a container
> image and deploy workflow ready but not yet deployed. Dense retrieval, generation and
> evaluation are not built yet; see the [roadmap](#roadmap). Data: GOV.UK only for now.
> This README only claims what exists.

**Live demo:** coming after the first Cloud Run deploy (runbook:
[docs/DEPLOY.md](docs/DEPLOY.md)). Not affiliated with or endorsed by the UK government; not
financial advice.

## Quickstart

```bash
make install     # venv + deps + git hooks
make check       # ruff, mypy --strict, pytest (coverage gate 85%)
make ingest      # fetch -> parse -> chunk  =>  data/processed/{chunks.jsonl,manifest.json}
make serve       # API on http://localhost:8080/docs  (GET /health, GET /search?q=...)
```

Dependencies are pinned in `requirements.lock` (hash-checked, used by the image) and
`requirements-dev.lock`. After editing `pyproject.toml`, run `make lock` and commit both.

Requires Python 3.11+. Before running against live sites, set a real contact address in
`USER_AGENT` (`src/ukmoney_rag/fetch.py`) or pass `--user-agent`.

## What ingestion does

```
configs/sources.toml ─► Fetcher (cache-first, robots-aware) ─► parse ─► chunk ─► chunks.jsonl
                              data/raw/<name>.{json,html}                       + manifest.json
```

| Decision | Why |
|---|---|
| GOV.UK via the **Content API** (JSON), not HTML scraping | Structured, stable, and intended for reuse; gives per-part deep links for citations. |
| Freshness = max(`public_updated_at`, `change_history`) | The API's `public_updated_at` can lag real changes (e.g. the State Pension rate change is only in `change_history`). Stale freshness would poison answers about rates. |
| Chunk **within sections**; keep lists/tables whole; glue headings and lead-in sentences to what follows | Retrieval quality depends on chunks that make sense alone. |
| Content-addressed `chunk_id` | Unchanged text keeps its ID, so later stages can skip re-embedding. |
| Output excludes timestamps; timestamps live in `manifest.json` | Same cached inputs + config ⇒ byte-identical `chunks.jsonl` (tested). |
| Fail loudly: unsupported schema, withdrawn page, <50 words extracted | A silent empty document is worse than a crash. |
| Never evade bot protection; unreadable `robots.txt` ⇒ refuse | Ethics and legal exposure; also honest engineering. |

## Data sources and licensing

All content comes from GOV.UK via the Content API, reused under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
Every chunk carries its source URL and licence, and API responses include an attribution and
no-endorsement notice, as the OGL requires. The HTML parser exists and is tested, but no HTML
sources are in the corpus. `data/` is gitignored because it is rebuilt from
`configs/sources.toml`.

## Current corpus (5 GOV.UK guides)

107 chunks · 5 documents · words/chunk min 12 / median 160 / max 180 (word count is a proxy for
tokens until the embedding model is chosen).

## Layout

```
src/ukmoney_rag/
  models.py   immutable pydantic contracts (Source, Section, Document, Chunk)
  fetch.py    cache-first HTTP: rate limit, retries, robots.txt, bot-wall detection
  parse.py    GOV.UK JSON / generic HTML -> Document
  chunk.py    structure-aware chunking
  ingest.py   CLI + manifest
  search.py   BM25 keyword retrieval (baseline)
  api.py      FastAPI service
tests/        unit tests (no network) + one opt-in live smoke test (`make test-live`)
configs/      corpus definition
docs/         DEPLOY.md (Cloud Run runbook)
Dockerfile    multi-stage, non-root, corpus baked in
```

## Known limitations

- 4 of 107 chunks are under 40 words (single-sentence section tails). Deliberately not tuned by
  eye; chunk parameters will be chosen against the retrieval eval set (Week 3).
- Word-count chunking, not tokenizer-based.
- Multi-column/complex tables are flattened to pipe-separated rows.
- Only GOV.UK `guide` and single-`body` formats are supported; others raise `ParseError`.
- BM25 cannot tell current from outdated guidance (e.g. a query about income tax rates also
  surfaces the *previous tax years* page). That goes into the eval set as a known failure.
- Each deploy re-fetches GOV.UK, so the corpus can change between deploys; `/health` exposes
  the corpus hash so any answer can be traced to its exact corpus.

## Roadmap

- [x] **Week 1** — repo, tooling, CI, ingestion, BM25 search API, Docker, Cloud Run CD
- [ ] **Week 2** — embeddings + vector index, dense retrieval, index versioning
- [ ] **Week 3** — hand-built eval set (questions + gold chunks), retrieval metrics, chunk-size ablation
- [ ] **Week 4** — grounded generation with citations, refusal when unsupported, answer-level eval
- [ ] **Week 5** — rate limiting/cost guards for the LLM endpoint, results write-up, failure analysis

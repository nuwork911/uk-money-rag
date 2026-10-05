# ADR 0001: Corpus sources

**Status:** accepted (Week 1; recorded retroactively in Week 2). **Revisit:** if a
source offers a sanctioned feed, or the eval shows coverage gaps.

## Decision
GOV.UK only, fetched via the Content API (`https://www.gov.uk/api/content/<path>`).
The served corpus is a committed snapshot (`data/snapshot/`, pinned by
`eval/CORPUS_SHA256`), not a live crawl at deploy time.

## Why
- **Licence.** GOV.UK content is published under the Open Government Licence v3.0, which
  permits reuse with attribution (see `NOTICE`).
- **Structured source.** The Content API returns JSON (`details.parts[]` for multi-part
  guides, `details.body` for single pages), so parsing doesn't depend on page layout.
- **Coverage.** Benefits, pensions and tax across 49 guides is enough to evaluate
  retrieval meaningfully.
- **Reproducibility.** A pinned snapshot means every deploy and eval run sees the same
  bytes, whatever GOV.UK serves that day.

## Details that matter
- The fetcher respects `robots.txt` and **fails closed** if it cannot be read.
- 401/403 responses and bot-challenge pages raise `BlockedError`: never cached, never
  retried around. Being blocked is treated as an answer, not an obstacle.

## Rejected or deferred
- **MoneyHelper (rejected).** It blocks automated access (403 and a challenge page, even
  on `robots.txt`). Getting around that would breach its terms; sanctioned reuse goes
  through its partnerships/syndication programme.
- **Citizens Advice (rejected).** Main-site content is "All rights reserved".
- **mygov.scot, nidirect.gov.uk (deferred).** Both are OGL v3.0 and accessible, so they are
  licensable. Deferred because nation-specific guidance can contradict GOV.UK answers, and
  handling that needs per-chunk jurisdiction tagging the eval doesn't yet cover.

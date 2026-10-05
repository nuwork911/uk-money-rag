# Retrieval eval set

Hand-written questions with gold answers, used to compare retrievers and chunking choices.
Everything here is pinned to the corpus in `eval/CORPUS_SHA256`.

| File | What it is |
|---|---|
| `cases.toml` | The eval set: question, tags, and where the answer is (URL + verbatim quote) |
| `known_failures.toml` | Failures seen before they could be measured; each points at eval cases |
| `CORPUS_SHA256` | sha256 of `data/snapshot/chunks.jsonl` the labels were written against |
| `results/` | Output of `make eval` (JSON per retriever + `summary.md`). Generated, never hand-edited |

## How a label works

A case names the page and a short quote (3-30 words) that answers the question. A chunk is
relevant if it comes from that URL and contains the quote (whitespace, curly quotes and case
are normalised). A case may list several `[[case.evidence]]` entries when the answer appears
in more than one place; any one of them counts.

Labels are quotes, not chunk IDs, on purpose: chunk IDs are content hashes and all change when
the chunker changes, so ID labels would make the chunk-size ablation impossible to score.

## Metrics

- **hit@k**: the answer is in the top k (recall@k for a single-answer question).
- **MRR@10**: mean of 1/rank of the first relevant chunk (0 if not in the top 10).
- **words in top 5**: how much text the retriever hands to a generator. Bigger chunks raise
  hit@k for free, so chunking comparisons report this alongside.
- **Paired bootstrap 95% CI** on the difference between retrievers, resampling cases. With
  ~40 cases, differences under ~0.1 MRR are usually inside the noise; the report says so.

## Writing cases (protocol)

1. **Write the question before opening the page**, in the words a real person would use.
   Questions copied from page text favour BM25 and inflate its score. Tag these `blind`.
2. Find the answer on the live GOV.UK page, then confirm the quote against the snapshot:
   `.venv/bin/ukmoney-eval find "<quote>" --url <page url>` (0 chunks = fix the quote).
3. Tag topic (`benefits`, `tax`, `pensions`, `savings`, `debt`, `work`) and wording
   (`lexical` if it shares key terms with the page, else `paraphrase`). Aim for at least a
   third `paraphrase`: that is where dense retrieval should earn its cost.
4. If the corpus cannot answer it, set `answerable = false` and give no evidence. These are
   not scored for retrieval; Week 4's `/ask` must decline them.
5. Run `make eval-validate`. CI runs the same check on every PR.

Do not look at retriever output while labelling. After a run, audit every case that *both*
retrievers miss: it is either a hard question (keep it) or a label that's too narrow (add the
other place the answer appears, and note it in `notes`). Never edit a label to make a
retriever look better.

## Seed cases

The ten `seed` cases were written while reading the corpus (to cover the known failures and
show the format), so they may favour lexical retrieval. Results are also reported for `blind`
cases alone.

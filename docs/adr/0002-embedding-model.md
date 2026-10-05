# ADR 0002: Embedding model

**Status:** accepted (Week 2). **Revisit:** after the Week 3 eval.

## Decision
`BAAI/bge-small-en-v1.5`, served via `fastembed` 0.8.x (quantised ONNX export,
`Qdrant/bge-small-en-v1.5-onnx-Q`), 384-d, L2-normalised, cosine similarity.

## Why
- **Image size / cold start.** ONNX Runtime instead of PyTorch keeps the runtime image small and
  avoids a GPU-oriented dependency tree on a CPU-only, scale-to-zero service.
- **Fits the data.** GOV.UK chunks are ≤ ~180 words, well inside the model's 512-token limit.
- **Good enough to measure against.** A small, well-known retrieval model is a fair baseline; the
  Week 3 eval decides whether `bge-base` (768-d, ~3x larger) is worth it.

## Details that matter
- BGE v1.5 uses a query-side instruction. fastembed does **not** add it for this model, so
  `embed.py` adds it explicitly; it is part of `index_id` because changing it changes every query.
- Passages are embedded as `embed_text` (`title > section\n\ntext`), not raw `text`.
- The ONNX file's sha256 is pinned. A silent upstream re-upload fails the build instead of
  producing a subtly different index.

## Rejected
- `sentence-transformers` + PyTorch: much heavier image for no quality gain at this size.
- Hosted embedding APIs: per-call cost, a secret to manage, and a network hop on every query.

## Observed: embeddings are not byte-stable across machines (Week 2)

| Environment                                   | CPU                 | `embeddings_sha256` |
|-----------------------------------------------|---------------------|---------------------|
| Laptop, WSL2 `.venv`                          | AMD Ryzen 5 4500U   | `3048faee2f64…`     |
| Docker image build, same laptop               | AMD Ryzen 5 4500U   | `3048faee2f64…`     |
| Separate Linux sandbox used to verify the code | not recorded        | `f8a1c6…`           |

All three produced the same `index_id` (`ab2eb5d42c02`) from the same snapshot, with the same
pinned model (`51f1bd0a…`) and fastembed 0.8.1. Scores differed in the 4th decimal; ranking was
unchanged on the queries compared.

**Interpretation.** Embedding is deterministic for a fixed machine and environment (two separate
builds on the laptop are bit-identical). The cross-machine difference is most likely
CPU-dependent ONNX Runtime kernels; not isolated, since the sandbox's software environment was
not fully controlled.

**Consequences.**
- `index_id` identifies the *inputs*; `embeddings_sha256` identifies the *bytes*. Only the
  former is expected to be stable, which is why indexes are keyed by inputs.
- The served index is built inside the image, which pins software but not hardware: the build
  host's CPU can differ from Cloud Run's, so query and passage embeddings may differ at ~1e-4.
  Accepted.
- `make docker-smoke` logs `index_id`, `embeddings_sha256` and CPU model on every CI run, so
  this record extends itself.
- The evaluation harness compares rankings and metrics, never raw scores or hashes.

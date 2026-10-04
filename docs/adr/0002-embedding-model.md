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

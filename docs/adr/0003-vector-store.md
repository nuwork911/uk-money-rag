# ADR 0003: Vector store

**Status:** accepted (Week 2).

## Decision
Exact brute-force cosine search over a NumPy `float32` matrix, persisted as a versioned
directory (`indexes/<index_id>/`) and baked into the container image.

## Why
- At ~10^2–10^4 chunks, one matrix–vector product takes well under a millisecond. Exact search
  has recall 1.0 by definition, so it is also the ground truth any ANN index must be checked
  against.
- No extra service to run, secure, pay for, or keep in sync. The index is immutable and ships
  with the code that reads it.

## When this stops being right
- Corpus beyond roughly 10^5–10^6 chunks (memory and latency), **or**
- the corpus must update without a redeploy, **or**
- metadata filtering inside the search is needed at scale.

Then: FAISS/hnswlib (in-process ANN), or a managed store (pgvector, Qdrant). The `DenseIndex`
API (`load`, `search`) is the seam; measure ANN recall@k against this exact index before switching.

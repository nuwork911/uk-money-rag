"""Retrieval metrics. Pure functions over ranked chunk IDs; no I/O, no retriever knowledge.

Each eval case has one answer, which may live in several chunks (chunk overlap, or the same
sentence on more than one page). Any one of them is enough for a RAG system, so:

- ``hit_at_k``: 1 if any relevant chunk is in the top k, else 0. This is recall@k for a
  single-answer question; "recall over all relevant chunks" would penalise a retriever for not
  returning the duplicate copy, which a generator never needs.
- ``reciprocal_rank``: 1 / rank of the first relevant chunk, 0 if none is in the top k.
- ``paired_bootstrap``: CI for mean(b - a) over the same cases. Resampling cases (not runs) is
  the right unit: the eval set is the sample, and both retrievers are deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence, Set
from dataclasses import dataclass

import numpy as np


def first_relevant_rank(ranked: Sequence[str], relevant: Set[str]) -> int | None:
    """1-based rank of the first relevant ID, or None if absent."""
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return rank
    return None


def _check(relevant: Set[str], k: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant:
        raise ValueError("metric is undefined for a case with no relevant chunks")


def hit_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    _check(relevant, k)
    return 1.0 if first_relevant_rank(ranked[:k], relevant) is not None else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    _check(relevant, k)
    rank = first_relevant_rank(ranked[:k], relevant)
    return 0.0 if rank is None else 1.0 / rank


@dataclass(frozen=True)
class BootstrapCI:
    mean_diff: float
    low: float
    high: float
    n_cases: int
    n_boot: int
    seed: int

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> BootstrapCI:
    """Percentile bootstrap CI for mean(b) - mean(a), resampling the same cases for both."""
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty, equal-length score lists (same cases, same order)")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    diffs = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    tail = (1 - level) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return BootstrapCI(
        mean_diff=float(diffs.mean()),
        low=float(low),
        high=float(high),
        n_cases=len(diffs),
        n_boot=n_boot,
        seed=seed,
    )

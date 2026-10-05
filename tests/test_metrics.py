from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ukmoney_rag.metrics import (
    first_relevant_rank,
    hit_at_k,
    paired_bootstrap,
    reciprocal_rank,
)

RANKED = ["a", "b", "c", "d"]


def test_first_relevant_rank_is_one_based() -> None:
    assert first_relevant_rank(RANKED, {"c"}) == 3
    assert first_relevant_rank(RANKED, {"z"}) is None


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 1.0), (4, 1.0)])
def test_hit_at_k(k: int, expected: float) -> None:
    assert hit_at_k(RANKED, {"b", "d"}, k) == expected


def test_reciprocal_rank_uses_first_relevant_and_cutoff() -> None:
    assert reciprocal_rank(RANKED, {"b", "d"}, 10) == 0.5
    assert reciprocal_rank(RANKED, {"d"}, 3) == 0.0  # beyond the cutoff counts as a miss


@pytest.mark.parametrize("fn", [hit_at_k, reciprocal_rank])
def test_metrics_reject_bad_input(fn: object) -> None:
    assert callable(fn)
    with pytest.raises(ValueError, match="k must be"):
        fn(RANKED, {"a"}, 0)
    with pytest.raises(ValueError, match="no relevant"):
        fn(RANKED, set(), 3)


ids = st.lists(st.sampled_from("abcdefghij"), unique=True, max_size=10)


@given(ranked=ids, relevant=st.sets(st.sampled_from("abcdefghij"), min_size=1))
def test_metric_invariants(ranked: list[str], relevant: set[str]) -> None:
    hits = [hit_at_k(ranked, relevant, k) for k in range(1, 11)]
    assert hits == sorted(hits)  # monotone in k
    rr = reciprocal_rank(ranked, relevant, 10)
    assert rr == 0.0 or 1 / rr == first_relevant_rank(ranked, relevant)
    assert (rr > 0) == (hits[-1] == 1.0)


def test_bootstrap_is_seeded_and_brackets_the_mean() -> None:
    a = [0.0, 0.5, 0.0, 0.25]
    b = [0.5, 1.0, 0.25, 0.5]  # b beats a on every case
    ci = paired_bootstrap(a, b, n_boot=2000, seed=7)
    assert ci == paired_bootstrap(a, b, n_boot=2000, seed=7)
    assert ci.low <= ci.mean_diff <= ci.high
    assert ci.mean_diff == pytest.approx(0.375)
    assert ci.excludes_zero


def test_bootstrap_identical_runs_give_zero_width_interval() -> None:
    ci = paired_bootstrap([0.2, 0.8], [0.2, 0.8])
    assert (ci.low, ci.mean_diff, ci.high) == (0.0, 0.0, 0.0)
    assert not ci.excludes_zero


@pytest.mark.parametrize(
    ("a", "b", "level", "message"),
    [
        ([], [], 0.95, "non-empty"),
        ([1.0], [1.0, 0.0], 0.95, "equal-length"),
        ([1.0], [1.0], 1.0, "level"),
    ],
)
def test_bootstrap_rejects_bad_input(
    a: list[float], b: list[float], level: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_bootstrap(a, b, level=level)

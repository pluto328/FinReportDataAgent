"""Score normalization and weighted fusion across retrieval channels."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def min_max_normalize_scores(scores: list[float]) -> list[float]:
    """Map raw channel scores to [0, 1] within a single batch."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    span = max_s - min_s
    if span <= 0:
        return [1.0] * len(scores)
    return [(s - min_s) / span for s in scores]


def fuse_weighted_batches(
    batches: list[list[T]],
    weights: list[float],
    *,
    key_fn: Callable[[T], str],
    score_fn: Callable[[T], float],
    set_score: Callable[[T, float], T],
    copy_fn: Callable[[T], T],
) -> list[T]:
    """Min-max normalize each channel batch, then weighted-sum merge by key."""
    merged: dict[str, T] = {}
    for batch, weight in zip(batches, weights):
        if not batch or weight <= 0:
            continue
        raw = [score_fn(item) for item in batch]
        normalized = min_max_normalize_scores(raw)
        for item, norm_score in zip(batch, normalized):
            key = key_fn(item)
            weighted = norm_score * weight
            if key in merged:
                prev = merged[key]
                merged[key] = set_score(copy_fn(prev), score_fn(prev) + weighted)
            else:
                copy = copy_fn(item)
                merged[key] = set_score(copy, weighted)
    return sorted(merged.values(), key=score_fn, reverse=True)

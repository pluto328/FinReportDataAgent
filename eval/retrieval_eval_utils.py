"""Shared helpers for retrieval weight evaluation (eval-only, no app changes)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def min_max_normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    span = max_s - min_s
    if span <= 0:
        return [1.0] * len(scores)
    return [(s - min_s) / span for s in scores]


def gold_norm_in_batch(hits: list[dict[str, Any]], gold_chunk_id: str) -> float:
    """Min-max normalized score of gold chunk within a single-channel top-k batch (0 if absent)."""
    if not hits:
        return 0.0
    raw = [float(h["score"]) for h in hits]
    norms = min_max_normalize_scores(raw)
    for hit, norm in zip(hits, norms):
        if str(hit["chunk_id"]) == gold_chunk_id:
            return norm
    return 0.0


def fuse_channel_hits(
    es_hits: list[dict[str, Any]],
    dense_hits: list[dict[str, Any]],
    *,
    dense_weight: float,
) -> list[tuple[str, float]]:
    """Min-max normalize each channel batch, then weighted-sum merge (same as online fusion)."""
    es_weight = 1.0 - dense_weight
    merged: dict[str, float] = {}

    for batch, weight in ((es_hits, es_weight), (dense_hits, dense_weight)):
        if not batch or weight <= 0:
            continue
        raw = [float(h["score"]) for h in batch]
        norms = min_max_normalize_scores(raw)
        for hit, norm in zip(batch, norms):
            cid = str(hit["chunk_id"])
            merged[cid] = merged.get(cid, 0.0) + norm * weight

    return sorted(merged.items(), key=lambda kv: kv[1], reverse=True)


def hits_from_es_raw(raw_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": str(h.get("chunk_id", h.get("id", ""))),
            "score": float(h.get("_score", 0.0)),
        }
        for h in raw_hits
    ]


def hits_from_dense_raw(raw_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": str(h.get("id", "")),
            "score": float(h.get("_score", 0.0)),
        }
        for h in raw_hits
    ]


def load_latest_testset(result_dir: Path, pattern: str = "testset_*.json") -> tuple[Path, list[dict]]:
    files = sorted(result_dir.glob(pattern))
    if not files:
        return Path(), []
    path = files[-1]
    data = __import__("json").loads(path.read_text(encoding="utf-8"))
    return path, list(data.get("samples") or [])


def dense_weights() -> list[float]:
    return [round(i / 10, 1) for i in range(1, 10)]

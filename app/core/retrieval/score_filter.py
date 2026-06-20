"""Embedding cosine similarity helpers (diagnostics / scripts)."""

from __future__ import annotations

from app.infrastructure.embedding_service import EmbeddingService


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


async def embed_cosine_scores(
    embed: EmbeddingService,
    query: str,
    texts: list[str],
) -> list[float]:
    if not texts:
        return []
    vectors = await embed.embed([query, *texts])
    qv = vectors[0]
    return [cosine_similarity(qv, tv) for tv in vectors[1:]]

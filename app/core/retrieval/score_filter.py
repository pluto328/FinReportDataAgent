"""Cosine-similarity filtering for retrieval results (bge-m3 normalized embeddings)."""

from __future__ import annotations

from app.infrastructure.embedding_service import EmbeddingService
from app.schemas.document import ScoredChunk
from app.schemas.structured import ScoredMetaRecord


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


async def filter_chunks_by_min_score(
    embed: EmbeddingService,
    query: str,
    chunks: list[ScoredChunk],
    min_score: float,
) -> list[ScoredChunk]:
    if not chunks or min_score <= 0:
        return chunks
    texts = [c.chunk.content for c in chunks]
    scores = await embed_cosine_scores(embed, query, texts)
    out: list[ScoredChunk] = []
    for chunk, sim in zip(chunks, scores):
        if sim >= min_score:
            item = chunk.model_copy(deep=True)
            item.score = sim
            out.append(item)
    return out


async def filter_meta_by_min_score(
    embed: EmbeddingService,
    query: str,
    hits: list[ScoredMetaRecord],
    min_score: float,
) -> list[ScoredMetaRecord]:
    if not hits or min_score <= 0:
        return hits
    texts = [h.record.search_text or h.record.file_name for h in hits]
    scores = await embed_cosine_scores(embed, query, texts)
    out: list[ScoredMetaRecord] = []
    for hit, sim in zip(hits, scores):
        if sim >= min_score:
            item = hit.model_copy(deep=True)
            item.score = sim
            out.append(item)
    return out

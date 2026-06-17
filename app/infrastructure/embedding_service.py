"""Embedding and rerank model singleton."""

from __future__ import annotations

import asyncio
import gc
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from app.common.logger import logger
from app.config.settings import Settings, get_settings
from app.infrastructure.hf_hub_config import configure_hf_hub

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

_LOG_PATH = Path(__file__).resolve().parents[2] / "debug-72ff74.log"
_SESSION = "72ff74"
_MIN_TORCH = (2, 6)
_EMBED_BATCH_SIZE = 8


def _torch_version_tuple() -> tuple[int, ...] | None:
    try:
        import torch  # noqa: PLC0415

        parts: list[int] = []
        base = torch.__version__.split("+")[0].split("-")[0]
        for piece in base.split("."):
            if piece.isdigit():
                parts.append(int(piece))
        return tuple(parts) if parts else None
    except Exception:
        return None


def _require_torch_for_hub() -> str:
    version = _torch_version_tuple()
    if version is None:
        return "unknown"
    if version < _MIN_TORCH:
        msg = (
            f"torch {'.'.join(map(str, version))} is too old for transformers model loading "
            f"(need >= {_MIN_TORCH[0]}.{_MIN_TORCH[1]}). "
            "Run: pip install \"torch>=2.6.0\""
        )
        raise RuntimeError(msg)
    return ".".join(map(str, version))


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # region agent log
    payload = {
        "sessionId": _SESSION,
        "runId": "embed-load",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    # endregion


@lru_cache
def _embed_model(name: str, device: str) -> SentenceTransformer:
    settings = get_settings()
    configure_hf_hub(settings)
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    torch_version = "unknown"
    try:
        torch_version = _require_torch_for_hub()
    except RuntimeError as exc:
        _agent_log(
            "H6",
            "embedding_service.py:_embed_model",
            "load_failed",
            {"model": name, "error": str(exc)[:400]},
        )
        raise

    logger.info(
        "loading embed model {} on {} (first run may download weights, please wait…)",
        name,
        device,
    )
    _agent_log(
        "H6",
        "embedding_service.py:_embed_model",
        "load_start",
        {
            "model": name,
            "device": device,
            "torch_version": torch_version,
            "HF_HOME": __import__("os").environ.get("HF_HOME", ""),
        },
    )
    try:
        model = SentenceTransformer(name, device=device)
    except Exception as exc:
        _agent_log(
            "H6",
            "embedding_service.py:_embed_model",
            "load_failed",
            {"model": name, "error": str(exc)[:400]},
        )
        raise
    max_len = getattr(model, "max_seq_length", None)
    logger.info("embed model {} ready (max_seq_length={})", name, max_len)
    _agent_log(
        "H6",
        "embedding_service.py:_embed_model",
        "load_ok",
        {"model": name, "max_seq_length": max_len},
    )
    return model


@lru_cache
def _rerank_model(name: str, device: str) -> CrossEncoder:
    configure_hf_hub(get_settings())
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    logger.info("loading rerank model {} on {} (please wait…)", name, device)
    model = CrossEncoder(name, device=device)
    logger.info("rerank model {} ready", name)
    return model


class EmbeddingService:
    """Lazy-loaded embedding and rerank."""

    def __init__(self, settings: Settings | None = None) -> None:
        s = settings or get_settings()
        self._embed_name = s.embed_model_name
        self._rerank_name = s.rerank_model_name
        self._device = s.device

    async def warmup(self) -> None:
        """Load embed/rerank models off the event loop (startup or first request)."""
        await asyncio.to_thread(_embed_model, self._embed_name, self._device)
        await asyncio.to_thread(_rerank_model, self._rerank_name, self._device)
        logger.info("embedding models warmed up")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = await asyncio.to_thread(_embed_model, self._embed_name, self._device)
        total = len(texts)
        if total > 1:
            logger.info("embedding {} chunk(s) with {}…", total, self._embed_name)

        vectors: list[list[float]] = []
        batch_count = (total + _EMBED_BATCH_SIZE - 1) // _EMBED_BATCH_SIZE
        for start in range(0, total, _EMBED_BATCH_SIZE):
            batch = texts[start : start + _EMBED_BATCH_SIZE]
            batch_idx = start // _EMBED_BATCH_SIZE + 1
            if batch_count > 1:
                logger.info(
                    "embedding sub-batch {}/{} ({} chunk(s))…",
                    batch_idx,
                    batch_count,
                    len(batch),
                )

            def _run(items: list[str] = batch) -> list[list[float]]:
                encoded = model.encode(
                    items,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return encoded.tolist()

            vectors.extend(await asyncio.to_thread(_run))

        if total > 1:
            logger.info("embedded {} vector(s)", len(vectors))
        gc.collect()
        return vectors

    async def embed_query(self, query: str) -> list[float]:
        vectors = await self.embed([query])
        return vectors[0]

    async def similarity(self, query: str, text: str) -> float:
        from app.core.retrieval.score_filter import cosine_similarity

        qv, tv = await self.embed([query, text])
        return cosine_similarity(qv, tv)

    async def rerank_score(self, query: str, text: str) -> float:
        ranked = await self.rerank(query, [text], top_k=1)
        return float(ranked[0][1]) if ranked else 0.0

    async def rerank(self, query: str, docs: list[str], top_k: int) -> list[tuple[int, float]]:
        if not docs:
            return []
        logger.info("reranking {} candidate(s) with {}…", len(docs), self._rerank_name)
        model = await asyncio.to_thread(_rerank_model, self._rerank_name, self._device)
        pairs = [[query, d] for d in docs]

        def _run() -> list[tuple[int, float]]:
            scores = model.predict(pairs)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]

        return await asyncio.to_thread(_run)

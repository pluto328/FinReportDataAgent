"""Run ablation evaluation against a generated testset."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.logger import logger, setup_logger
from app.config.settings import Settings, get_settings
from app.core.retrieval.bm25_retriever import BM25Retriever
from app.core.retrieval.dense_retriever import DenseRetriever
from app.core.retrieval.ensemble import EnsembleRetriever
from app.core.retrieval.es_retriever import ESRetriever
from app.core.retrieval.reranker import Reranker
from app.infrastructure.embedding_service import EmbeddingService
from app.infrastructure.es_client import ESClient
from app.infrastructure.vector_client import VectorClient
from app.schemas.metrics import EvalMetrics


SCHEMES: list[tuple[str, dict[str, bool | float]]] = [
    ("full_hybrid", {"enable_es": True, "enable_bm25": True, "enable_dense": True}),
    ("es_only", {"enable_es": True, "enable_bm25": False, "enable_dense": False}),
    ("bm25_only", {"enable_es": False, "enable_bm25": True, "enable_dense": False}),
    ("dense_only", {"enable_es": False, "enable_bm25": False, "enable_dense": True}),
]


def _load_latest_testset(result_dir: Path) -> list[dict]:
    files = sorted(result_dir.glob("testset_*.json"))
    if not files:
        return []
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    return data.get("samples", [])


def _recall_at_k(expected: str, hits: list, k: int) -> float:
    top = hits[:k]
    for h in top:
        source = getattr(getattr(h, "chunk", h), "source_file", "") or ""
        doc_id = getattr(getattr(h, "chunk", h), "doc_id", "") or ""
        if expected in source or expected in doc_id:
            return 1.0
    return 0.0


def _apply_scheme(settings: Settings, overrides: dict[str, bool | float]) -> Settings:
    data = settings.model_dump()
    data.update(overrides)
    return Settings.model_validate(data)


async def _eval_scheme(
    settings: Settings,
    samples: list[dict],
) -> EvalMetrics:
    es = ESClient(settings)
    vectors = VectorClient(settings)
    embed = EmbeddingService(settings)
    retriever = EnsembleRetriever(
        settings,
        ESRetriever(es),
        BM25Retriever(es),
        DenseRetriever(vectors, embed),
        Reranker(embed),
    )
    recalls: list[float] = []
    start = time.perf_counter()
    try:
        for sample in samples:
            query = sample.get("query", "")
            expected = sample.get("expected_doc_id", "")
            if not query:
                continue
            hits = await retriever.search(query)
            recalls.append(_recall_at_k(expected, hits, settings.final_top_k))
    finally:
        await es.close()

    elapsed = time.perf_counter() - start
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    return EvalMetrics(
        scheme_name="",
        recall_at_k=mean_recall,
        precision_at_k=mean_recall,
        mrr=mean_recall,
        config={
            "enable_es": settings.enable_es,
            "enable_bm25": settings.enable_bm25,
            "enable_dense": settings.enable_dense,
            "elapsed_sec": round(elapsed, 3),
        },
    )


async def main() -> None:
    base = get_settings()
    base.ensure_directories()
    setup_logger(log_dir=str(base.log_dir), level=base.log_level)
    samples = _load_latest_testset(base.eval_result_path)
    if not samples:
        logger.error("no testset found under {}; run gen_testset.py first", base.eval_result_path)
        return

    results: list[EvalMetrics] = []
    for name, overrides in SCHEMES:
        settings = _apply_scheme(base, overrides)
        metrics = await _eval_scheme(settings, samples)
        metrics.scheme_name = name
        results.append(metrics)
        logger.info("{} recall@k={:.3f}", name, metrics.recall_at_k)

    out_dir = base.eval_result_path
    out_file = out_dir / f"metrics_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(
        json.dumps([m.model_dump() for m in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("wrote {}", out_file)


if __name__ == "__main__":
    asyncio.run(main())

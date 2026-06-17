"""ES + BM25 + Dense 三路混合检索与精排。

子模块规约：
- base.py：BaseRetriever，统一 async def search(query, top_k) -> list[ScoredChunk]
- es/bm25/dense_retriever.py：各负责单路召回，不做融合
- ensemble.py：读 ENABLE_* 动态 asyncio.gather；权重 ES_WEIGHT/BM25_WEIGHT/DENSE_WEIGHT；融合后交 reranker
- reranker.py：仅用 RERANK_MODEL_NAME 精排，不做融合

性能：三路检索优先并发，注意 ES/Chroma 连接池。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "BaseRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "EnsembleRetriever",
    "ESRetriever",
    "MetaDenseRetriever",
    "MetaEnsembleRetriever",
    "MetaKeywordRetriever",
    "Reranker",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "BaseRetriever": ("app.core.retrieval.base", "BaseRetriever"),
    "ESRetriever": ("app.core.retrieval.es_retriever", "ESRetriever"),
    "BM25Retriever": ("app.core.retrieval.bm25_retriever", "BM25Retriever"),
    "DenseRetriever": ("app.core.retrieval.dense_retriever", "DenseRetriever"),
    "Reranker": ("app.core.retrieval.reranker", "Reranker"),
    "EnsembleRetriever": ("app.core.retrieval.ensemble", "EnsembleRetriever"),
    "MetaKeywordRetriever": ("app.core.retrieval.meta_keyword_retriever", "MetaKeywordRetriever"),
    "MetaDenseRetriever": ("app.core.retrieval.meta_dense_retriever", "MetaDenseRetriever"),
    "MetaEnsembleRetriever": ("app.core.retrieval.meta_ensemble", "MetaEnsembleRetriever"),
}

if TYPE_CHECKING:
    from app.core.retrieval.base import BaseRetriever
    from app.core.retrieval.bm25_retriever import BM25Retriever
    from app.core.retrieval.dense_retriever import DenseRetriever
    from app.core.retrieval.ensemble import EnsembleRetriever
    from app.core.retrieval.es_retriever import ESRetriever
    from app.core.retrieval.meta_dense_retriever import MetaDenseRetriever
    from app.core.retrieval.meta_ensemble import MetaEnsembleRetriever
    from app.core.retrieval.meta_keyword_retriever import MetaKeywordRetriever
    from app.core.retrieval.reranker import Reranker


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module_path, attr = _EXPORTS[name]
    module = __import__(module_path, fromlist=[attr])
    return getattr(module, attr)

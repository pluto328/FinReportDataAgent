"""ES keyword + dense vector hybrid retrieval and reranking.

Submodules:
- es_retriever.py: Elasticsearch keyword recall
- dense_retriever.py: Chroma vector similarity recall
- ensemble.py: fuses ES + dense, then cross-encoder rerank + score filter
- meta_*: structured metadata keyword + dense fusion with rerank
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "BaseRetriever",
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
    "DenseRetriever": ("app.core.retrieval.dense_retriever", "DenseRetriever"),
    "Reranker": ("app.core.retrieval.reranker", "Reranker"),
    "EnsembleRetriever": ("app.core.retrieval.ensemble", "EnsembleRetriever"),
    "MetaKeywordRetriever": ("app.core.retrieval.meta_keyword_retriever", "MetaKeywordRetriever"),
    "MetaDenseRetriever": ("app.core.retrieval.meta_dense_retriever", "MetaDenseRetriever"),
    "MetaEnsembleRetriever": ("app.core.retrieval.meta_ensemble", "MetaEnsembleRetriever"),
}

if TYPE_CHECKING:
    from app.core.retrieval.base import BaseRetriever
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

"""底层连接封装，参数取自 settings，无业务、无 Prompt。

子模块规约：
- es_client.py：ES 8.x + Basic Auth；索引 mapping 与 DocChunk 一致
- vector_client.py：Chroma 持久化；失效分片只更新 metadata，不删向量文件
- llm_client.py：langchain-openai 兼容端点；温度等取自 settings

性能：
- Embedding/Rerank（torch）按 DEVICE 加载，逻辑集中本层或 retrieval，不散落
- Redis（ENABLE_REDIS=true）仅可选缓存，须有降级路径
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["ESClient", "EmbeddingService", "LLMClient", "VectorClient"]

_EXPORTS: dict[str, tuple[str, str]] = {
    "ESClient": ("app.infrastructure.es_client", "ESClient"),
    "LLMClient": ("app.infrastructure.llm_client", "LLMClient"),
    "VectorClient": ("app.infrastructure.vector_client", "VectorClient"),
    "EmbeddingService": ("app.infrastructure.embedding_service", "EmbeddingService"),
}

if TYPE_CHECKING:
    from app.infrastructure.embedding_service import EmbeddingService
    from app.infrastructure.es_client import ESClient
    from app.infrastructure.llm_client import LLMClient
    from app.infrastructure.vector_client import VectorClient


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module_path, attr = _EXPORTS[name]
    module = __import__(module_path, fromlist=[attr])
    return getattr(module, attr)

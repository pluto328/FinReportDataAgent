"""Bootstrap ChromaDB without onnxruntime default embeddings.

Chroma 0.5.x evaluates DefaultEmbeddingFunction() at import time, which requires
onnxruntime. This project always passes embeddings from EmbeddingService, so we
inject a stub embedding_functions module before chromadb is imported.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

_PATCHED = "_rag_chroma_bootstrapped"


def _find_chroma_utils_path() -> list[str]:
    import site

    for base in (*site.getsitepackages(), site.getusersitepackages()):
        if not base:
            continue
        utils = Path(base) / "chromadb" / "utils"
        if utils.is_dir():
            return [str(utils)]
    return []


def _stub_default_embedding_function() -> Any:
    # Real EmbeddingFunction is created after chromadb import in vector_client.
    return None


def ensure_chroma_without_onnx() -> None:
    if sys.modules.get("chromadb.utils.embedding_functions") and getattr(
        sys.modules["chromadb.utils.embedding_functions"],
        _PATCHED,
        False,
    ):
        return

    stub_ef = types.ModuleType("chromadb.utils.embedding_functions")
    stub_ef.DefaultEmbeddingFunction = _stub_default_embedding_function
    setattr(stub_ef, _PATCHED, True)

    utils_mod = sys.modules.get("chromadb.utils")
    utils_path = _find_chroma_utils_path()
    if utils_mod is None:
        utils_mod = types.ModuleType("chromadb.utils")
        sys.modules["chromadb.utils"] = utils_mod
    if utils_path:
        utils_mod.__path__ = utils_path  # type: ignore[attr-defined]
    utils_mod.embedding_functions = stub_ef  # type: ignore[attr-defined]

    sys.modules["chromadb.utils.embedding_functions"] = stub_ef

"""Retrieval abstract base."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.document import ScoredChunk
from app.schemas.structured import ScoredMetaRecord


class BaseRetriever(ABC):
    name: str = "base"

    @abstractmethod
    async def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Search text chunks."""


class BaseMetaRetriever(ABC):
    name: str = "meta_base"

    @abstractmethod
    async def search(self, query: str, top_k: int) -> list[ScoredMetaRecord]:
        """Search structured metadata records."""

"""Sliding-window text chunker."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

from app.config.settings import Settings
from app.schemas.document import AssetKind, DocChunk


class Chunker:
    def __init__(self, settings: Settings) -> None:
        self._size = settings.chunk_size
        self._overlap = settings.chunk_overlap

    def chunk_text(self, text: str, *, source_file: str, doc_id: str, md5: str) -> list[DocChunk]:
        chunks: list[DocChunk] = []
        if not text.strip():
            return chunks
        start = 0
        idx = 0
        while start < len(text):
            end = start + self._size
            content = text[start:end]
            chunk_id = hashlib.md5(f"{doc_id}:{idx}".encode()).hexdigest()
            chunks.append(
                DocChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    content=content,
                    asset_kind=AssetKind.TEXT,
                    md5=md5,
                    source_file=source_file,
                    chunk_index=idx,
                )
            )
            idx += 1
            if end >= len(text):
                break
            start = end - self._overlap
        return chunks

    @staticmethod
    def file_md5(path: Path) -> str:
        h = hashlib.md5()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()

    @staticmethod
    def new_doc_id(path: Path) -> str:
        return hashlib.md5(str(path.resolve()).encode()).hexdigest()[:16] or uuid4().hex[:16]

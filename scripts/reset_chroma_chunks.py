"""Reset Chroma rag_collection (upsert hang / corruption recovery)."""

from __future__ import annotations

import time

from app.config.settings import get_settings
from app.infrastructure.chroma_bootstrap import ensure_chroma_without_onnx

ensure_chroma_without_onnx()
import chromadb
from chromadb.config import Settings as ChromaSettings

from pathlib import Path

s = get_settings()
settings = ChromaSettings(
    anonymized_telemetry=False,
    chroma_product_telemetry_impl="app.infrastructure.chroma_telemetry.NoOpProductTelemetry",
)

if s.chroma_use_http:
    client = chromadb.HttpClient(
        host=s.chroma_http_host,
        port=s.chroma_http_port,
        settings=settings,
    )
    print(f"Chroma HTTP → {s.chroma_http_host}:{s.chroma_http_port}", flush=True)
else:
    client = chromadb.PersistentClient(path=str(s.vector_persist_path), settings=settings)
    print(f"Chroma local → {s.vector_persist_path}", flush=True)

name = s.chroma_collection
try:
    client.delete_collection(name)
    print(f"deleted collection {name}", flush=True)
except Exception as exc:
    print(f"delete skipped: {exc}", flush=True)

col = client.get_or_create_collection(name)
print(f"recreated {name} count={col.count()}", flush=True)
t0 = time.perf_counter()
col.upsert(
    ids=["reset-smoke-1"],
    embeddings=[[0.01] * 1024],
    documents=["smoke"],
    metadatas=[{"doc_id": "t", "source_file": "t.pdf", "status": "online"}],
)
print(f"smoke upsert ok in {time.perf_counter() - t0:.2f}s count={col.count()}", flush=True)

cache_dir = Path(s.cache_path)
removed = 0
if cache_dir.is_dir():
    for entry in cache_dir.glob("*.json"):
        entry.unlink(missing_ok=True)
        removed += 1
print(f"cleared {removed} parsed_cache entr{'y' if removed == 1 else 'ies'}", flush=True)
print("Restart a single uvicorn instance to re-ingest.", flush=True)

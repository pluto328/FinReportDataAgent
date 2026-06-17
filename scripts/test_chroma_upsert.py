"""Quick Chroma upsert diagnostic (run while uvicorn is stopped)."""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from app.config.settings import Settings, get_settings
from app.infrastructure.vector_client import VectorClient


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("count", nargs="?", type=int, default=8)
    parser.add_argument("--temp", action="store_true", help="use isolated Chroma path under data/chroma_diag_tmp")
    args = parser.parse_args()
    n = args.count
    if args.temp:
        tmp = Path("data/chroma_diag_tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        s = Settings(vector_persist_path=tmp, chroma_use_http=False)
        print(f"Chroma local path: {s.vector_persist_path}", flush=True)
    else:
        s = get_settings()
        if s.chroma_use_http:
            print(f"Chroma HTTP: {s.chroma_http_host}:{s.chroma_http_port}", flush=True)
        else:
            print(f"Chroma local path: {s.vector_persist_path}", flush=True)
    v = VectorClient(s)
    dim = 1024
    ids = [f"diag-chunk-{i}" for i in range(n)]
    embs = [[0.01] * dim for _ in range(n)]
    docs = ["sample text " * 100 for _ in range(n)]
    metas = [{"doc_id": "diag", "source_file": "t.pdf", "status": "online"} for _ in range(n)]
    t0 = time.perf_counter()
    print(f"upsert starting n={n}...", flush=True)
    await v.upsert_chunks_batch(ids, embs, docs, metas)
    print(f"upsert done in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

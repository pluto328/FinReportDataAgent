"""Generate evaluation queries from raw documents via LLM."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.logger import logger, setup_logger
from app.config.settings import get_settings
from app.core.ingestion.parser import TEXT_EXTENSIONS, DocumentParser
from app.infrastructure.llm_client import LLMClient


async def _generate_for_file(llm: LLMClient, doc_id: str, text: str) -> list[dict]:
    snippet = text[:4000]
    prompt = (
        f"文档编号:{doc_id}\n"
        "根据以下文档内容生成 3 条中文问答测试样本，输出 JSON 数组："
        '[{"query":"...","expected_doc_id":"..."}]\n'
        f"正文:\n{snippet}"
    )
    raw = await llm.ainvoke(prompt)
    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        data = json.loads(raw[start:end]) if start >= 0 and end > start else []
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        logger.warning("failed to parse LLM output for {}", doc_id)
    return []


async def main() -> None:
    settings = get_settings()
    settings.ensure_directories()
    setup_logger(log_dir=str(settings.log_dir), level=settings.log_level)
    out_dir = settings.eval_result_path
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = DocumentParser()
    llm = LLMClient(settings)
    samples: list[dict] = []

    for path in sorted(settings.raw_doc_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        doc_id = path.stem
        text = await parser.parse(path)
        rows = await _generate_for_file(llm, doc_id, text)
        for row in rows:
            samples.append(
                {
                    "query": row.get("query", ""),
                    "expected_doc_id": row.get("expected_doc_id", doc_id),
                    "source_file": str(path),
                }
            )
        logger.info("gen_testset {} -> {} samples", path.name, len(rows))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(samples),
        "samples": samples,
    }
    out_file = out_dir / f"testset_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote {}", out_file)


if __name__ == "__main__":
    asyncio.run(main())

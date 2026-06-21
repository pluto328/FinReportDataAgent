"""Report-phase read processed data tool."""

from __future__ import annotations

from typing import Any

from app.config.settings import get_settings
from app.core.tools.artifact_utils import load_artifact_text
from app.core.tools.base_tool import BaseTool


class ReadDataFileTool(BaseTool):
    name = "read_data_file"
    description = (
        "读取会话中间数据或检索到的数据文件全文。"
        "入参：path（文件名，从中间数据中选取）。"
        "返回：content。"
    )

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        path = str(kwargs.get("path", ""))
        if not path:
            return {"ok": False, "error": "path is required"}
        try:
            max_chars = get_settings().context_size_threshold_chars
            text = load_artifact_text(path, max_chars=max_chars)
            return {
                "ok": True,
                "path": path,
                "content": text,
                "char_count": len(text),
                "truncated": len(text) >= max_chars,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

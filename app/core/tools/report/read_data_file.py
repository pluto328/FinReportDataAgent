"""Report-phase read processed data tool."""

from __future__ import annotations

from typing import Any

from app.core.tools.artifact_utils import load_artifact_text
from app.core.tools.base_tool import BaseTool


class ReadDataFileTool(BaseTool):
    name = "read_data_file"
    description = (
        "读取会话中间数据或检索到的数据文件全文。"
        "入参：path（填 catalog 中的文件名，如 liability_top5.csv；系统自动解析为绝对路径）。"
        "返回：content。"
    )

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        path = str(kwargs.get("path", ""))
        if not path:
            return {"ok": False, "error": "path is required"}
        try:
            text = load_artifact_text(path)
            return {"ok": True, "path": path, "content": text}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

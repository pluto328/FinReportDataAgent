"""Report-phase read processed data tool."""

from __future__ import annotations

from typing import Any

from app.core.tools.artifact_utils import load_artifact_text
from app.core.tools.base_tool import BaseTool


class ReadDataFileTool(BaseTool):
    name = "read_data_file"
    description = (
        "读取数据文件的全部内容。"
        "入参：path。"
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

"""Preview read tool — read structured file preview only."""

from __future__ import annotations

from typing import Any

from app.core.tools.base_tool import BaseTool
from app.core.tools.structured_ops import read_table_preview


class PreviewReadTool(BaseTool):
    name = "preview_read"
    description = (
        "读取结构化数据文件的预览。"
        "入参：file_path。"
        "返回：preview。"
    )

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        path = str(kwargs["file_path"])
        df = read_table_preview(path)
        return {"preview": df.head(20).to_dict(orient="records")}

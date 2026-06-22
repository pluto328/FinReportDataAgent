"""Preview read tool — read structured file preview only."""

from __future__ import annotations

from typing import Any

from app.core.tools.base_tool import BaseTool
from app.core.tools.structured_ops import normalize_file_paths


class PreviewReadTool(BaseTool):
    name = "preview_read"
    description = (
        "读取结构化数据文件的预览（每个文件最多20行，写入会话 state）。"
        "入参 file_paths（绝对路径列表，可同时预览多个检索到的原始数据文件）；"
        "仅预览单个文件时也可填 file_path。"
        "路径须来自「检索到的数据文件」，不能是尚未生成的中间产物。"
        "返回：paths（已预览路径列表）；预览内容不在工具结果中返回。"
    )

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        paths = normalize_file_paths(
            kwargs.get("file_path"),
            file_paths=kwargs.get("file_paths"),
        )
        if not paths:
            return {"error": "preview_read 缺少 file_path 或 file_paths"}
        return {"paths": paths, "path": paths[0]}

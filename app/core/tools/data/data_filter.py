"""Data filter tool — filter and save in original format."""

from __future__ import annotations

from typing import Any

from app.config.settings import Settings, get_settings
from app.core.tools.artifact_utils import save_dataframe_processed
from app.core.tools.base_tool import BaseTool
from app.core.tools.structured_ops import filter_rows, read_table_full


class DataFilterTool(BaseTool):
    name = "data_filter"
    description = (
        "按列条件筛选结构化数据并保存为原格式文件（文件名以 _processed 加原后缀结尾）。"
        "入参：file_path（绝对路径）、column（列名）、op（eq|contains，默认 eq）、value（比较值）。"
        "返回：path（保存后的绝对路径）。"
    )

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        path = str(kwargs["file_path"])
        session_id = str(kwargs.get("session_id", "default"))
        settings: Settings = kwargs.get("settings") or get_settings()
        df = read_table_full(path)
        filtered = filter_rows(
            df,
            str(kwargs["column"]),
            str(kwargs.get("op", "eq")),
            str(kwargs.get("value", "")),
        )
        ref = save_dataframe_processed(filtered, path, session_id, settings, mode="tool")
        return {"path": ref.path}

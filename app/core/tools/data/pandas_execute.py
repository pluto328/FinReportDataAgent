"""Pandas code execution tool."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.config.settings import Settings, get_settings
from app.core.tools.artifact_utils import save_dataframe_processed
from app.core.tools.base_tool import BaseTool
from app.core.tools.structured_ops import read_table_full


class PandasExecuteTool(BaseTool):
    name = "pandas_execute"
    description = (
        "对结构化数据执行 pandas 代码，结果 DataFrame 保存为 _processed 加原后缀的文件。"
        "入参：file_path（绝对路径）、code（Python 代码，使用 df/pd，结果赋给 result 或 df）。"
        "返回：path（保存后的绝对路径）；非 DataFrame 结果返回 error。"
    )

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        file_path = str(kwargs["file_path"])
        code = str(kwargs.get("code", ""))
        session_id = str(kwargs.get("session_id", "default"))
        settings: Settings = kwargs.get("settings") or get_settings()
        if not code.strip():
            return {"error": "code is required", "error_code": "missing_code"}
        try:
            df = read_table_full(file_path)
            local_ns: dict[str, Any] = {"df": df, "pd": pd}
            exec(code, {"__builtins__": {}}, local_ns)  # noqa: S102
            result = local_ns.get("result", local_ns.get("df", df))
            if isinstance(result, pd.DataFrame):
                ref = save_dataframe_processed(result, file_path, session_id, settings, mode="tool")
                return {"path": ref.path}
            return {"error": "result must be a DataFrame", "error_code": "invalid_result"}
        except Exception as exc:
            return {"error": str(exc), "error_code": type(exc).__name__}

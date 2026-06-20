"""Pandas code execution tool."""

from __future__ import annotations

import builtins
import re
from typing import Any

import numpy as np
import pandas as pd

from app.config.settings import Settings, get_settings
from app.core.tools.artifact_utils import preview_dataframe_rows, save_dataframe_processed
from app.core.tools.base_tool import BaseTool
from app.core.tools.structured_ops import read_table_full

_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "len",
        "list",
        "map",
        "max",
        "min",
        "range",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "isinstance",
        "True",
        "False",
        "None",
    )
}


def _sanitize_pandas_code(code: str) -> str:
    """Drop import/read lines; df/pd/np are injected by the tool runtime."""
    kept: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        if re.search(r"\bpd\.read_\w+\s*\(", stripped):
            continue
        if re.search(r"\bdf\s*=\s*pd\.read_\w+\s*\(", stripped):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


class PandasExecuteTool(BaseTool):
    name = "pandas_execute"
    description = (
        "对结构化数据执行 pandas 代码，结果 DataFrame 保存为新文件。"
        "入参：file_path（绝对路径）、code（Python 代码，已预置 df/pd/np，禁止 import、pd.read_* 与任何注释，"
        "artifact_name（必填，保存文件名含后缀，根据描述取名，不与已有中间数据文件名重复）、"
        "artifact_description（必填，产物中文说明，如「负债榜前五名数据」）。"
        "返回：path（保存后的绝对路径）、preview（产物前3行）；非 DataFrame 结果返回 error。"
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
            sanitized = _sanitize_pandas_code(code)
            if not sanitized:
                return {"error": "code is empty after sanitization", "error_code": "empty_code"}
            local_ns: dict[str, Any] = {"df": df, "pd": pd, "np": np}
            exec(sanitized, {"__builtins__": _SAFE_BUILTINS}, local_ns)  # noqa: S102
            result = local_ns.get("result", local_ns.get("df", df))
            if isinstance(result, pd.DataFrame):
                ref = save_dataframe_processed(
                    result, file_path, session_id, settings, mode="tool",
                    artifact_name=str(kwargs.get("artifact_name", "")),
                )
                return {"path": ref.path, "preview": preview_dataframe_rows(result)}
            return {"error": "result must be a DataFrame", "error_code": "invalid_result"}
        except Exception as exc:
            return {"error": str(exc), "error_code": type(exc).__name__}

"""Pandas code execution tool."""

from __future__ import annotations

import builtins
import re
from typing import Any

import numpy as np
import pandas as pd

from app.config.settings import Settings, get_settings
from app.core.tools.artifact_utils import save_dataframe_processed
from app.core.tools.base_tool import BaseTool
from app.core.tools.structured_ops import normalize_file_paths, read_table_full

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


def _pandas_df_name(index: int) -> str:
    """第1个文件 df，第2个 df1，第3个 df2 …（与常见 pandas 习惯一致）。"""
    return "df" if index == 0 else f"df{index}"


def _inject_pandas_legacy_df_aliases(local_ns: dict[str, Any], file_count: int) -> None:
    """兼容旧 prompt 中的 df2/df3 命名（第2个文件曾写作 df2）。"""
    if file_count >= 2 and "df1" in local_ns:
        local_ns.setdefault("df2", local_ns["df1"])
    if file_count >= 3 and "df2" in local_ns:
        local_ns.setdefault("df3", local_ns["df2"])


PANDAS_MULTI_FILE_VAR_RULE = (
    "pandas 多文件变量（已注入 pd/np，禁止 import、注释、pd.read_*）："
    "第1个文件→df，第2个→df1，第3个→df2，依此类推；"
    "单文件只用 df；代码末尾须 result=...（DataFrame）。"
)


class PandasExecuteTool(BaseTool):
    name = "pandas_execute"
    description = (
        "对结构化数据执行 pandas 代码，可同时处理多个文件，结果 DataFrame 保存为新文件。"
        "入参：file_paths（绝对路径列表，可含多个检索到的原始数据文件）、"
        "code（Python 代码，已预置 df/df1/df2…/pd/np，第一个文件为 df，第二个 df1，以此类推；"
        "禁止 import、pd.read_* 与任何注释）、"
        "artifact_name（必填，保存文件名含后缀，根据描述取名，不与已有中间数据文件名重复）、"
        "artifact_description（必填，产物中文说明，如「负债榜前五名数据」）。"
    )

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        paths = normalize_file_paths(
            kwargs.get("file_path"),
            file_paths=kwargs.get("file_paths"),
        )
        code = str(kwargs.get("code", ""))
        session_id = str(kwargs.get("session_id", "default"))
        settings: Settings = kwargs.get("settings") or get_settings()
        if not paths:
            return {"error": "file_paths is required", "error_code": "missing_file_paths"}
        if not code.strip():
            return {"error": "code is required", "error_code": "missing_code"}
        try:
            local_ns: dict[str, Any] = {"pd": pd, "np": np}
            for i, fp in enumerate(paths):
                local_ns[_pandas_df_name(i)] = read_table_full(fp)
            _inject_pandas_legacy_df_aliases(local_ns, len(paths))
            sanitized = _sanitize_pandas_code(code)
            if not sanitized:
                return {"error": "code is empty after sanitization", "error_code": "empty_code"}
            exec(sanitized, {"__builtins__": _SAFE_BUILTINS}, local_ns)  # noqa: S102
            result = local_ns.get("result", local_ns.get("df", local_ns.get(_pandas_df_name(0))))
            if isinstance(result, pd.DataFrame):
                ref = save_dataframe_processed(
                    result,
                    paths[0],
                    session_id,
                    settings,
                    mode="tool",
                    artifact_name=str(kwargs.get("artifact_name", "")),
                )
                return {"path": ref.path}
            return {"error": "result must be a DataFrame", "error_code": "invalid_result"}
        except Exception as exc:
            return {"error": str(exc), "error_code": type(exc).__name__}

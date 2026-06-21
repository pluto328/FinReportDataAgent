"""SQL execution tool (DuckDB read-only SELECT)."""

from __future__ import annotations

import re
from typing import Any

from app.config.settings import Settings, get_settings
from app.core.tools.artifact_utils import preview_dataframe_rows, save_dataframe_processed
from app.core.tools.base_tool import BaseTool
from app.core.tools.structured_ops import execute_sql_on_files, normalize_file_paths


def _sanitize_sql(sql: str) -> str:
    """Remove SQL comments; LLM sql must be comment-free."""
    text = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    lines: list[str] = []
    for line in text.splitlines():
        if "--" in line:
            line = line.split("--", 1)[0]
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return " ".join(lines).strip()


class SqlExecuteTool(BaseTool):
    name = "sql_execute"
    description = (
        "对结构化文件执行只读 SELECT（DuckDB），可同时处理多个文件，结果保存为新文件。"
        "入参：file_paths（绝对路径列表，可含多个检索到的原始数据文件）、"
        "sql（SELECT 语句，禁止任何注释 -- 或 /* */；多文件时第一张表名为 src，第二张 src2，第三张 src3，以此类推）、"
        "artifact_name（必填，保存文件名含后缀，根据描述取名，不与已有中间数据文件名重复）、"
        "artifact_description（必填，产物中文说明，如「某指标汇总表」）。"
        "返回：path（保存后的绝对路径）、preview（产物前3行）。"
    )

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        paths = normalize_file_paths(
            kwargs.get("file_path"),
            file_paths=kwargs.get("file_paths"),
        )
        sql = _sanitize_sql(str(kwargs["sql"]))
        session_id = str(kwargs.get("session_id", "default"))
        settings: Settings = kwargs.get("settings") or get_settings()
        if not paths:
            return {"error": "file_paths is required", "error_code": "missing_file_paths"}
        if not sql:
            return {"error": "sql is empty after sanitization", "error_code": "empty_sql"}
        try:
            df = execute_sql_on_files(paths, sql)
            ref = save_dataframe_processed(
                df,
                paths[0],
                session_id,
                settings,
                mode="sql",
                artifact_name=str(kwargs.get("artifact_name", "")),
            )
            return {"path": ref.path, "preview": preview_dataframe_rows(df)}
        except Exception as exc:
            return {"error": str(exc), "error_code": type(exc).__name__}

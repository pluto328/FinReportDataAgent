"""SQL execution tool (DuckDB read-only SELECT)."""

from __future__ import annotations

import re
from typing import Any

from app.config.settings import Settings, get_settings
from app.core.tools.artifact_utils import save_dataframe_processed
from app.core.tools.base_tool import BaseTool
from app.core.tools.structured_ops import execute_sql_on_file


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
        "对结构化文件执行只读 SELECT（DuckDB），结果保存为 _processed 加原后缀的文件。"
        "入参：file_path（绝对路径）、sql（SELECT 语句，禁止任何注释 -- 或 /* */）、"
        "artifact_description（必填，产物中文说明，如「某指标汇总表」）。"
        "返回：path（保存后的绝对路径）。"
    )

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        file_path = str(kwargs["file_path"])
        sql = _sanitize_sql(str(kwargs["sql"]))
        session_id = str(kwargs.get("session_id", "default"))
        settings: Settings = kwargs.get("settings") or get_settings()
        if not sql:
            return {"error": "sql is empty after sanitization", "error_code": "empty_sql"}
        try:
            df = execute_sql_on_file(file_path, sql)
            ref = save_dataframe_processed(df, file_path, session_id, settings, mode="sql")
            return {"path": ref.path}
        except Exception as exc:
            return {"error": str(exc), "error_code": type(exc).__name__}

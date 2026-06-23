"""Plan tool: load session history and build follow-up context."""

from __future__ import annotations

from typing import Any

from app.core.session.history_store import build_history_context
from app.core.tools.base_tool import BaseTool


class LoadHistoryContextTool(BaseTool):
    name = "load_history_context"
    description = (
        "加载本会话全部历史问题与回答摘要，供规划器判断如何结合前文。"
        "入参：无（由节点自动注入 session_id）。"
        "返回：turn_count、history（列表）、context_text（全部历史拼接文本）。"
    )

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        session_id = str(kwargs.get("session_id", "default"))
        settings = kwargs.get("settings")
        chat_history = kwargs.get("chat_history") or []

        payload = build_history_context(
            session_id,
            settings=settings,
            chat_history=chat_history,
        )
        return {"method": self.name, **payload}

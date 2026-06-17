"""Plan tool: load session history and build follow-up context."""

from __future__ import annotations

from typing import Any

from app.core.session.history_store import format_all_turns, load_session_turns
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

        turns = load_session_turns(session_id, settings)
        history: list[dict[str, Any]] = [
            {
                "turn_id": t.turn_id,
                "question": t.question,
                "answer_summary": t.answer_summary,
            }
            for t in turns
        ]
        context_text = format_all_turns(turns)

        if not history and chat_history:
            user_msgs = [m for m in chat_history if m.get("role") == "user"]
            asst_msgs = [m for m in chat_history if m.get("role") in ("assistant", "ai")]
            pairs: list[dict[str, Any]] = []
            for i, u in enumerate(user_msgs):
                q = str(u.get("content", ""))
                a = ""
                if i < len(asst_msgs):
                    a = str(asst_msgs[i].get("content", ""))[:300]
                pairs.append({"turn_id": i + 1, "question": q, "answer_summary": a})
            if pairs:
                history = pairs
                context_text = "\n\n".join(
                    f"第{p['turn_id']}轮\n【历史问题】{p['question']}\n【历史回答摘要】{p['answer_summary']}"
                    for p in pairs
                )

        return {
            "method": self.name,
            "turn_count": len(history),
            "history": history,
            "context_text": context_text,
        }

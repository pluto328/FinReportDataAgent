"""Persist and load per-session Q&A turns (question + answer summary)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config.settings import Settings, get_settings
from app.schemas.session import SessionTurnRecord

_FOLLOW_UP_HINTS = (
    "刚才",
    "之前",
    "上面",
    "上述",
    "继续",
    "那么",
    "还有",
    "同样",
    "基于此",
    "在此基础上",
    "刚才的",
    "前面的",
    "它",
    "这个",
    "那个",
)


def looks_like_follow_up(query: str) -> bool:
    q = query.strip()
    if not q:
        return False
    return any(h in q for h in _FOLLOW_UP_HINTS)


def _history_path(session_id: str, settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    return s.cache_path / session_id / "session_history.json"


def load_session_turns(session_id: str, settings: Settings | None = None) -> list[SessionTurnRecord]:
    path = _history_path(session_id, settings)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return []
        return [SessionTurnRecord.model_validate(item) for item in raw]
    except (json.JSONDecodeError, OSError, ValueError):
        return []


def append_session_turn(
    record: SessionTurnRecord,
    session_id: str,
    settings: Settings | None = None,
) -> SessionTurnRecord:
    s = settings or get_settings()
    path = _history_path(session_id, s)
    path.parent.mkdir(parents=True, exist_ok=True)
    turns = load_session_turns(session_id, s)
    record.turn_id = len(turns) + 1
    turns.append(record)
    payload = [t.model_dump(mode="json") for t in turns]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def format_turn_context(turn: SessionTurnRecord) -> str:
    return (
        f"【历史问题】{turn.question}\n"
        f"【历史回答摘要】{turn.answer_summary}"
    )


def format_all_turns(turns: list[SessionTurnRecord]) -> str:
    if not turns:
        return "（无历史记录）"
    return "\n\n".join(f"第{t.turn_id}轮\n{format_turn_context(t)}" for t in turns)


def truncate_context_tail(text: str, max_chars: int) -> str:
    if not text or max_chars <= 0:
        return text or ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def build_history_context(
    session_id: str,
    *,
    settings: Settings | None = None,
    chat_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build session history payload (same shape as load_history_context tool)."""
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
        "turn_count": len(history),
        "history": history,
        "context_text": context_text,
    }


def format_planner_history_context(
    session_id: str,
    *,
    settings: Settings | None = None,
    chat_history: list[dict[str, Any]] | None = None,
    max_chars: int = 800,
) -> str:
    """Format history for planner prompt; keep tail ``max_chars`` when truncated."""
    payload = build_history_context(session_id, settings=settings, chat_history=chat_history)
    context_text = str(payload.get("context_text") or "")
    if not context_text or context_text == "（无历史记录）":
        return "（无历史记录）"
    truncated = truncate_context_tail(context_text, max_chars)
    if len(context_text) > max_chars:
        return f"（已截断，仅保留最近 {max_chars} 字）\n{truncated}"
    return truncated

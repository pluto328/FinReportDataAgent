"""Persist and load per-session Q&A turns (question + answer summary)."""

from __future__ import annotations

import json
from pathlib import Path

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

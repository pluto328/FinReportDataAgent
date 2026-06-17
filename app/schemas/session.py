"""Session Q&A turn record for follow-up context."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SessionTurnRecord(BaseModel):
    turn_id: int
    question: str
    answer_summary: str = ""
    report_mode: bool = False
    markdown_path: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)

"""Rule-based query checks; LLM entry validation is done in planner_node."""

from __future__ import annotations

import re

GUIDANCE_MESSAGE = """请输入与金融、研报或企业数据相关的问题。你可以问我例如：

• 某行业最近有哪些 AI 相关业绩突破？
• 帮我汇总某 CSV 里的销售数据并画趋势图
• 刚才的结论里，增长最快的是哪一块？
• 生成一份关于半导体板块的分析报告"""

INSUFFICIENT_DATA_MESSAGE = "数据不足，请联系管理员补充数据。"


def normalize_query(query: str) -> str:
    return (query or "").strip()


def is_empty_query(query: str) -> bool:
    return not normalize_query(query)


def looks_meaningless_by_rule(query: str) -> bool:
    text = normalize_query(query)
    if not text:
        return True
    if len(text) == 1:
        return True
    if len(text) <= 2 and not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text):
        return True
    if re.fullmatch(r"[\W_]+", text, flags=re.UNICODE):
        return True
    if re.fullmatch(r"(.)\1{2,}", text):
        return True
    return False

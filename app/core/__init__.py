"""核心业务：入库、检索、Agent、Tool 编排。

禁止：直接读 .env、初始化全局客户端（由 dependencies 注入）。

子包：ingestion / retrieval / agent / tools
"""

from __future__ import annotations

__all__ = ["agent", "ingestion", "retrieval", "tools"]

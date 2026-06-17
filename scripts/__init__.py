"""运维脚本：独立入口，不 import core 内部实现。

CLI 模块：
- monitor.py：watchdog 监听 RAW_DOC_PATH，防抖后触发 updater 增量入库
"""

from __future__ import annotations

__all__ = ["monitor"]

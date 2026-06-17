"""全局配置：.env → pydantic-settings。

settings.py 规约：
- 字段与 .env.example 全部 34 项一一对应
- 路径用 pathlib.Path；布尔/数值类型明确
- 路径规范化见 paths.py（跨平台防误解析、禁止逃逸 project_root）
- 提供 get_settings() 缓存单例（lru_cache 或 lifespan 注入）
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["Settings", "PathManager", "PROJECT_ROOT", "get_settings", "normalize_config_path"]

if TYPE_CHECKING:
    from app.config.paths import PathManager
    from app.config.settings import Settings as Settings
    from app.config.settings import get_settings as get_settings


def __getattr__(name: str) -> Any:
    if name in ("PathManager", "PROJECT_ROOT", "normalize_config_path"):
        from app.config import paths

        return getattr(paths, name)
    if name in ("Settings", "get_settings"):
        from app.config import settings

        return getattr(settings, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

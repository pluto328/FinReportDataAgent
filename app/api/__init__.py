"""REST 路由层：参数校验 + Depends 转发，不含业务逻辑。

子模块与前缀：
- doc_api.py   → /documents   文档上传/删除/刷新
- search_api.py → /search 知识库问答
- report_api.py → /report 结构化报告

规约：
- 上传用 UploadFile；异常映射 HTTPException + 统一错误体
- 实例全部 Depends 注入，禁止直接 new 客户端
"""

from __future__ import annotations

from fastapi import APIRouter

api_router = APIRouter()


def mount_routes(root: APIRouter | None = None) -> APIRouter:
    """Attach sub-routers; call from app.main during startup."""
    target = root or api_router

    from app.api.doc_api import router as doc_router
    from app.api.report_api import router as report_router
    from app.api.search_api import router as search_router
    from app.api.session_api import router as session_router

    target.include_router(doc_router, prefix="/documents", tags=["documents"])
    target.include_router(search_router, prefix="/search", tags=["search"])
    target.include_router(report_router, prefix="/report", tags=["report"])
    target.include_router(session_router, prefix="/session", tags=["session"])
    return target


__all__ = ["api_router", "mount_routes"]

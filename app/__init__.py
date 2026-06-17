"""Knowledge RAG System Agent 应用根包。

分层：api → dependencies → core → infrastructure；schemas/config/common 横切共享。

根目录模块规约：
- dependencies.py：Depends 组装 ES/向量/检索/LLM/Agent 单例与工厂，不含业务判断
- main.py：lifespan 启动时 sync_all()、关闭时释放客户端；挂载 CORS（CORS_ORIGINS）与聚合路由
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]

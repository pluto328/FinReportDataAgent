"""RAG 评测：独立入口，只调 HTTP 或公开 facade，不侵入 core 内部。

CLI 模块：
- gen_testset.py：读 RAW_DOC_PATH，LLM 生成 query，输出到 EVAL_RESULT_PATH
- rag_metric_eval.py：切换 env/参数做消融评测，禁止硬编码检索组合；指标用 RAGAS
"""
# 待拓展
# 基础指标：准确率、召回率、F1、语义相似度、检索命中率、响应耗时
# 业务指标：报告正确率、代码执行成功率、沙箱报错率、二次检索触发率
# 附加信息：评估批次、样本 ID、模型版本、评估时间、备注说明

from __future__ import annotations

__all__ = ["gen_testset", "rag_metric_eval"]

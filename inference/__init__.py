"""快速流式推理（fast inference SSE）模块入口。

子模块按职责拆分：
- ``config``：默认参数与开关
- ``redis_stream``：单 key 快照读写门面，含进程内 per-task 锁与聚合字段重算
- ``llm_stream``：LLM 流式 SSE 解析（``inference/llm_stream.py``，待实现）
- ``preview``/``skills_runner``/``react_loop``/``pipeline``：四阶段编排
- ``retrieval/``：BM25 + 语义 + RRF 混合检索
"""

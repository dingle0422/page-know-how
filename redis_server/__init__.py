"""轻量级 Redis 风格服务：KV（带 TTL） + List（FIFO 队列）。

用途：为本项目的异步化推理接口提供任务队列与结果存储后端；
也可作为通用的小型 KV/队列服务独立运行。

模块入口：
    - redis_server.run         单文件入口（FastAPI app + 路由 + main），直接执行
    - redis_server.client      Python 异步客户端（httpx）
    - redis_server.storage     内存存储引擎（KV + List + 快照）
    - redis_server.models      HTTP 接口 pydantic 模型
"""

__version__ = "0.1.0"

"""任务队列与结果存储后端（Redis Sentinel 客户端）。

用途：为本项目的异步化推理接口提供任务队列与结果存储后端。

后端说明：``client`` 直连专业 Redis Sentinel 集群（哨兵 + 主从）；
``RedisServerClient`` 公开 API 保持不变，调用方无需改动。

模块入口：
    - redis_server.client      Python 异步客户端（直连 Redis Sentinel 集群）
"""

__version__ = "0.1.0"

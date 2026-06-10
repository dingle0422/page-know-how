# redis_server

本目录现在只保留 **Redis Sentinel 集群异步客户端**：`redis_server.client`。

早期自研的 FastAPI 轻量服务端（`run.py` / `storage.py` / `models.py`）已清理，
线上链路统一直连 Redis Sentinel，不再经过 HTTP 网关中转。

---

## 快速开始

```bash
pip install -r requirements.txt
```

环境变量（可选，不配则使用代码内默认值）：

- `REDIS_SENTINELS`: `host:port,host:port,...`
- `REDIS_MASTER_NAME`: Sentinel 监控的 master 名称
- `REDIS_PASSWORD`: Redis / Sentinel 密码
- `REDIS_KEY_PREFIX`: 全局 key 前缀（默认空）

---

## Python 异步客户端

```python
from redis_server.client import RedisServerClient

async with RedisServerClient() as cli:
    # KV
    await cli.set("task:abc", {"status": "pending"}, ttl_seconds=86400)
    value, exists = await cli.get("task:abc")

    # Queue
    await cli.rpush("queue:reason:pending", "abc")
    task_id, ok = await cli.blpop("queue:reason:pending", timeout_seconds=30)
```

公开 API（保持历史兼容）：

- KV: `set/get/delete/exists/expire/ttl/keys`
- Queue: `rpush/lpop/blpop/llen/lrange/lrem`
- Admin: `health/stats/snapshot/flushall`

---

## 说明

- 调用侧（`app.py` / `task_queue.py` / `inference/redis_stream.py`）无需改动；
- `snapshot()` 在 Sentinel 模式下为 no-op（持久化由 Redis 集群管理）；
- `flushall()` 默认禁用，需显式设置 `REDIS_ALLOW_FLUSH=1` 才会执行清库。

# redis_server

一个**独立运行**的极简 Redis 风格服务，基于 `FastAPI + uvicorn` 纯 Python 实现，不依赖真实 Redis。

用途：为本项目的异步化推理接口 (`app.py /api/reason`) 提供

- **任务队列**（FIFO，支持阻塞弹出）：生产者 push 任务、Worker blpop 拉任务；
- **任务结果存储**（KV + TTL）：提交后立即生成 `task_id`，Worker 完成后写回结果，客户端按 `task_id` 轮询。

也可当作通用的 KV / 队列小服务独立用。

---

## 快速开始

```bash
pip install -r redis_server/requirements.txt

# 推荐姿势（三种等价，任选）
python -m redis_server.run            # 在项目根执行，把 redis_server 当包
python redis_server/run.py            # 在项目根执行
cd redis_server && python run.py      # 进入目录后执行

# 指定参数
python redis_server/run.py --host 127.0.0.1 --port 6390
```

启动日志类似：

```
[redis_server] 启动，snapshot=.../redis_server/data/snapshot.json interval=30.0s
INFO:     Uvicorn running on http://0.0.0.0:5000
```

> 默认监听 `0.0.0.0:5000`。如果同机还跑着本项目的推理服务 `app.py`（也用 5000），需要用
> `--port` 改端口或配 `REDIS_SRV_PORT` 环境变量（例如 `REDIS_SRV_PORT=6390`）。

健康检查：

```bash
curl http://127.0.0.1:6390/health
```

---

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `REDIS_SRV_HOST` | `127.0.0.1` | 监听地址，`--host` 可覆盖 |
| `REDIS_SRV_PORT` | `5000` | 监听端口，`--port` 可覆盖；与本项目推理服务冲突时改成 `6390` 之类 |
| `REDIS_SRV_DATA_DIR` | `redis_server/data` | 快照目录 |
| `REDIS_SRV_SNAPSHOT_INTERVAL` | `30` | 周期快照秒数，`<=0` 关闭周期快照（仍会在退出时落盘一次） |
| `REDIS_SRV_SWEEP_INTERVAL` | `60` | 过期扫描周期（秒） |
| `REDIS_SRV_AUTH_TOKEN` | 空 | 若设置，则除 `/health` 外所有接口强制校验 `X-Auth-Token` header |
| `REDIS_SRV_LOG_LEVEL` | `info` | uvicorn 日志等级 |

---

## 协议速查

所有响应统一结构：

```json
{ "status_code": 200, "message": "success", "data": { ... } }
```

### KV

| 方法 | 路径 | 请求体 / 查询 | `data` 字段 |
|---|---|---|---|
| POST | `/kv/set` | `{key, value, ttl_seconds?}` | `{exists, value}` |
| GET | `/kv/get/{key}` | — | `{exists, value}` |
| DELETE | `/kv/del/{key}` | — | `{deleted}` |
| GET | `/kv/exists/{key}` | — | `{exists}` |
| POST | `/kv/expire` | `{key, ttl_seconds}` | `{applied}` |
| GET | `/kv/ttl/{key}` | — | `{ttl_seconds}` （-1=永不过期，null=不存在） |
| GET | `/kv/keys?prefix=...` | — | `{keys: [...]}` |

### Queue（FIFO）

| 方法 | 路径 | 请求体 / 查询 | `data` 字段 |
|---|---|---|---|
| POST | `/queue/rpush` | `{queue, value}` | `{length}` |
| POST | `/queue/lpop` | `{queue}` | `{ok, value}` |
| POST | `/queue/blpop` | `{queue, timeout_seconds}` | `{ok, value}` （阻塞等待） |
| GET | `/queue/llen/{queue}` | — | `{length}` |
| GET | `/queue/lrange/{queue}?start=&stop=` | — | `{items}` |
| POST | `/queue/lrem` | `{queue, value, count}` | `{removed}` |

### Admin

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康 + 概览（**不鉴权**） |
| GET | `/admin/stats` | 详细统计 |
| POST | `/admin/snapshot` | 立即落盘一次 |
| POST | `/admin/flushall` | 清空所有数据（慎用） |

---

## Python 异步客户端

```python
from redis_server.client import RedisServerClient

async with RedisServerClient("http://127.0.0.1:6390") as cli:
    # KV
    await cli.set("task:abc", {"status": "pending"}, ttl_seconds=86400)
    value, exists = await cli.get("task:abc")

    # Queue
    await cli.rpush("queue:reason:pending", "abc")
    task_id, ok = await cli.blpop("queue:reason:pending", timeout_seconds=30)
```

---

## 推荐的推理服务接入约定

> 不是硬性要求，只是约定，`redis_server` 本身是通用 KV+Queue。

| Key / Queue | 用途 | 备注 |
|---|---|---|
| `queue:reason:pending` | 待推理任务队列，value = `task_id`（字符串） | 生产者 `/api/reason/submit` 入队；Worker blpop 拉取 |
| `task:{task_id}` | 任务元信息 + 结果，value 为 JSON | TTL 建议 24h；字段建议至少含 `status`（pending/running/done/failed）、`request`、`result`、`enqueue_time`、`start_time`、`end_time`、`error` |

客户端交互：
1. `POST /api/reason/submit` → 生成 `task_id`，`SET task:{id}`（pending），`RPUSH queue:reason:pending`，立即返回 `task_id`；
2. `GET /api/reason/result/{task_id}` → `GET task:{id}`，按 `status` 决定继续轮询还是返回最终结果。

---

## 持久化说明

- 快照是**纯 JSON**，可直接 `cat` 查看/调试；
- 写文件采用「临时文件 + `os.replace`」原子替换，避免宕机半写；
- 启动时若 `snapshot.json` 不存在或损坏会从空状态启动（日志中会打 warning）；
- 适用场景：**任务队列 + 结果缓存**，可容忍「最后 N 秒数据丢失」。如需强持久化，请改用真实 Redis + AOF。

---

## 局限性（和真实 Redis 的差异）

- 单进程单线程 asyncio，不适合百万级 QPS；本项目推理场景每秒几十 QPS 绰绰有余；
- 只实现了 KV / List 两种结构，没有 Hash / Set / ZSet / Pub-Sub / Stream；
- 没有集群 / 副本；高可用通过「定时快照 + 外部备份快照文件」兜底；
- `blpop` 是单 queue 单 waiter，不支持多 queue 一次监听。

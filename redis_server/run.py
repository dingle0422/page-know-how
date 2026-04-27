"""redis_server 单文件入口：FastAPI app + 路由 + 启动一气呵成。

仿照本项目 `app.py` 的写法：顶部 `sys.path.insert` 后，同级模块
（storage / models）用裸 import 即可，不依赖"redis_server 能否被当作顶层包导入"。

启动姿势（任选）：
    python run.py                         # 在 redis_server/ 目录内执行
    python redis_server/run.py            # 在项目根执行
    python -m redis_server.run            # 在项目根执行（把 redis_server 当包）
    python run.py --host 127.0.0.1 --port 6390
"""

import sys
import os
import time
import logging
import argparse
from contextlib import asynccontextmanager
from pathlib import Path

# 与 app.py 一致：把"本文件所在目录"放到 sys.path 最前面，
# 这样同级的 storage.py / models.py 就能直接 import，无需关心启动 cwd。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import Depends, FastAPI, Header, HTTPException

from storage import Store
from models import (
    BPopReq,
    DeleteResp,
    ExistsResp,
    ExpireReq,
    ExpireResp,
    FlushResp,
    GetResp,
    KeysResp,
    LLenResp,
    LRangeResp,
    LRemReq,
    LRemResp,
    PopReq,
    PopResp,
    PushReq,
    PushResp,
    SetReq,
    SnapshotResp,
    StatsResp,
    TtlResp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("redis_server")


# ----------------------------------------------------------------- 配置读取


def _project_default_data_dir() -> Path:
    return Path(os.path.dirname(os.path.abspath(__file__))) / "data"


DATA_DIR = Path(os.environ.get("REDIS_SRV_DATA_DIR") or _project_default_data_dir())
SNAPSHOT_INTERVAL = float(os.environ.get("REDIS_SRV_SNAPSHOT_INTERVAL", "30"))
SWEEP_INTERVAL = float(os.environ.get("REDIS_SRV_SWEEP_INTERVAL", "60"))
AUTH_TOKEN = os.environ.get("REDIS_SRV_AUTH_TOKEN") or ""

SNAPSHOT_FILE = DATA_DIR / "snapshot.json"


# ----------------------------------------------------------------- 全局实例

store = Store(
    snapshot_path=SNAPSHOT_FILE,
    snapshot_interval_seconds=SNAPSHOT_INTERVAL,
    sweep_interval_seconds=SWEEP_INTERVAL,
)


# --------------------------------------------------------------------- 鉴权


async def _require_auth(x_auth_token: str | None = Header(default=None)) -> None:
    """若配置了 REDIS_SRV_AUTH_TOKEN，则所有写/读接口强校验 header。"""
    if not AUTH_TOKEN:
        return
    if x_auth_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="invalid auth token")


# ------------------------------------------------------------------ lifespan


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"[redis_server] 启动，snapshot={SNAPSHOT_FILE} interval={SNAPSHOT_INTERVAL}s")
    await store.load_snapshot()
    await store.start_background_tasks()
    try:
        yield
    finally:
        logger.info("[redis_server] 停止，最后一次落盘…")
        await store.stop_background_tasks()
        try:
            await store.save_snapshot()
        except Exception as e:
            logger.exception(f"退出时落盘失败: {e}")


app = FastAPI(
    title="redis_server (lightweight KV + Queue)",
    version="0.1.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------------- Health

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "auth_required": bool(AUTH_TOKEN),
        "stats": await store.stats(),
    }


# PaaS / K8s readiness & liveness 探活；保持极简、零依赖（不访问 store），
# 避免任何内部故障把平台探活也一起带挂，造成无意义的反复重启。
# 与本项目 app.py 的 /example 语义一致。
@app.get("/example")
async def example():
    return {"status": "ok", "message": "redis_server is running."}


@app.get("/")
async def root_check():
    return {"status": "ok", "message": "redis_server is running."}


# ---------------------------------------------------------------------- KV


@app.post("/kv/set", response_model=GetResp, dependencies=[Depends(_require_auth)])
async def kv_set(req: SetReq):
    ttl = req.ttl_seconds if req.ttl_seconds and req.ttl_seconds > 0 else None
    await store.set(req.key, req.value, ttl_seconds=ttl)
    return GetResp(data={"exists": True, "value": req.value})


@app.get("/kv/get/{key}", response_model=GetResp, dependencies=[Depends(_require_auth)])
async def kv_get(key: str):
    value, exists = await store.get(key)
    return GetResp(data={"exists": exists, "value": value if exists else None})


@app.delete("/kv/del/{key}", response_model=DeleteResp, dependencies=[Depends(_require_auth)])
async def kv_delete(key: str):
    deleted = await store.delete(key)
    return DeleteResp(data={"deleted": deleted})


@app.get("/kv/exists/{key}", response_model=ExistsResp, dependencies=[Depends(_require_auth)])
async def kv_exists(key: str):
    return ExistsResp(data={"exists": await store.exists(key)})


@app.post("/kv/expire", response_model=ExpireResp, dependencies=[Depends(_require_auth)])
async def kv_expire(req: ExpireReq):
    ok = await store.expire(req.key, req.ttl_seconds)
    return ExpireResp(data={"applied": ok})


@app.get("/kv/ttl/{key}", response_model=TtlResp, dependencies=[Depends(_require_auth)])
async def kv_ttl(key: str):
    return TtlResp(data={"ttl_seconds": await store.ttl(key)})


@app.get("/kv/keys", response_model=KeysResp, dependencies=[Depends(_require_auth)])
async def kv_keys(prefix: str = ""):
    return KeysResp(data={"keys": await store.keys(prefix)})


# ------------------------------------------------------------------- Queue


@app.post("/queue/rpush", response_model=PushResp, dependencies=[Depends(_require_auth)])
async def queue_rpush(req: PushReq):
    length = await store.rpush(req.queue, req.value)
    return PushResp(data={"length": length})


@app.post("/queue/lpop", response_model=PopResp, dependencies=[Depends(_require_auth)])
async def queue_lpop(req: PopReq):
    value, ok = await store.lpop(req.queue)
    return PopResp(data={"ok": ok, "value": value if ok else None})


@app.post("/queue/blpop", response_model=PopResp, dependencies=[Depends(_require_auth)])
async def queue_blpop(req: BPopReq):
    """阻塞头出队；timeout_seconds<=0 表示一直等到有值。

    注意：实际等待时长受 HTTP 客户端 / 中间代理超时限制，建议不超过 60s。
    """
    value, ok = await store.blpop(req.queue, req.timeout_seconds)
    return PopResp(data={"ok": ok, "value": value if ok else None})


@app.get("/queue/llen/{queue}", response_model=LLenResp, dependencies=[Depends(_require_auth)])
async def queue_llen(queue: str):
    return LLenResp(data={"length": await store.llen(queue)})


@app.get("/queue/lrange/{queue}", response_model=LRangeResp, dependencies=[Depends(_require_auth)])
async def queue_lrange(queue: str, start: int = 0, stop: int = -1):
    return LRangeResp(data={"items": await store.lrange(queue, start, stop)})


@app.post("/queue/lrem", response_model=LRemResp, dependencies=[Depends(_require_auth)])
async def queue_lrem(req: LRemReq):
    removed = await store.lrem(req.queue, req.value, req.count)
    return LRemResp(data={"removed": removed})


# -------------------------------------------------------------------- Admin


@app.get("/admin/stats", response_model=StatsResp, dependencies=[Depends(_require_auth)])
async def admin_stats():
    return StatsResp(data=await store.stats())


@app.post("/admin/snapshot", response_model=SnapshotResp, dependencies=[Depends(_require_auth)])
async def admin_snapshot():
    path = await store.save_snapshot()
    return SnapshotResp(
        data={"path": str(path) if path else None, "saved_at": time.time()}
    )


@app.post("/admin/flushall", response_model=FlushResp, dependencies=[Depends(_require_auth)])
async def admin_flushall():
    """清空所有数据（仅限测试 / 紧急恢复场景使用）。"""
    await store.flushall()
    return FlushResp(data={"flushed": True})


# --------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight Redis-like KV + Queue server")
    parser.add_argument(
        "--host",
        default=os.environ.get("REDIS_SRV_HOST", "0.0.0.0"),
        help="监听地址（默认 0.0.0.0；本机开发可改为 127.0.0.1）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("REDIS_SRV_PORT", "5000")),
        help="监听端口（默认 5000；注意别和同机上的推理服务端口冲突）",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("REDIS_SRV_LOG_LEVEL", "info"),
        help="uvicorn 日志等级（默认 info）",
    )
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_keep_alive=600,
    )


if __name__ == "__main__":
    main()

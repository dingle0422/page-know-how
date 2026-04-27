"""异步推理任务队列：封装 redis_server 上 queue + KV 的业务语义。

职责划分
--------
1. 纯函数工具：submit_task / get_task / list_queued_tasks / list_running_tasks
   —— 供 app.py 的 submit / result / requestQueueStatus 路由直接调用。
2. `WorkerPool`：启动 N 个后台 asyncio worker，每个 worker 阻塞在
   BLPOP(`queue:reason:pending`) 上拉任务，回调注入的 executor 执行业务，
   并把状态/结果写回 `task:{task_id}`。

协议约定（必须与调用方一致）
-----------------------------
- 队列 key : ``queue:reason:pending``，存入内容为 task_id 字符串；
- 任务 key : ``task:{task_id}``，存入 JSON，字段见 ``submit_task`` 返回值
  （status / request / enqueue_time / start_time / end_time / result / error / worker_id）。

与老版本 ``app.py`` 内存 Semaphore 的对等关系
---------------------------------------------
- 老版本通过 asyncio.Semaphore(N) 限并发；本版本以"N 个 worker 各自串行拉"
  达到完全相同的并发上限（worker_count 对应 MAX_CONCURRENT_REASONING）；
- 老版本的 "queued" 状态 <=> 当前 BLPOP 还没 pop 到的 task_id 们；
- 老版本的 "running" 状态 <=> worker 已经 pop、但还没写 done/failed 的 task。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Awaitable, Callable, Optional

try:
    # 用于在 worker 循环里区分"网关瞬时错误（504/502/连接超时）"和真业务异常。
    from redis_server.client import RedisServerError
except Exception:
    # 降级：拿不到 RedisServerError 类型就让它退化成 Exception，循环仍然能跑。
    RedisServerError = Exception  # type: ignore[assignment]

import httpx


logger = logging.getLogger(__name__)


# 把"中间网关瞬时错误"归一化出来，便于 worker loop 单独处理：
# - RedisServerError 包了 httpx 之上的业务判断（含 504/502/503 等 HTTP 错误）；
# - httpx.TimeoutException 覆盖本地到网关这一段的网络超时；
# - httpx.TransportError 覆盖连接/读写阶段的各种链路错误（DNS、连接重置等）。
_TRANSIENT_NETWORK_ERRORS: tuple = (
    RedisServerError,
    httpx.TimeoutException,
    httpx.TransportError,
)


# ------------------------------------------------- 协议常量（redis_server 侧 key）

QUEUE_REASON_PENDING = "queue:reason:pending"
TASK_KEY_PREFIX = "task:"

# 任务状态枚举（字符串形式，便于直接 JSON 存取 / 跨语言调用方读取）
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

# 单条任务在 redis_server 上的默认寿命：24h。
# 客户端拿到 task_id 后在这段时间内轮询都能拿到结果；超过则需要重新提交。
DEFAULT_TASK_TTL_SECONDS = 24 * 3600


def _task_key(task_id: str) -> str:
    return f"{TASK_KEY_PREFIX}{task_id}"


def _now() -> float:
    return time.time()


# ------------------------------------------------------------- 纯函数操作层


async def submit_task(
    client,
    request_payload: dict,
    *,
    ttl_seconds: float = DEFAULT_TASK_TTL_SECONDS,
) -> dict:
    """生成 task_id、写入 task:{id}、RPUSH 到 queue:reason:pending。

    写 KV 再 push 队列的顺序不能颠倒：worker BLPOP 拿到 task_id 后会立刻
    GET task:{id}，若此时 KV 里还没有该条 record，会被 worker 记 warning 并
    直接丢弃（见 ``WorkerPool._run_one``）。
    """
    task_id = str(uuid.uuid4())
    record = {
        "task_id": task_id,
        "status": STATUS_PENDING,
        "request": request_payload,
        "enqueue_time": _now(),
        "start_time": None,
        "end_time": None,
        "result": None,
        "error": None,
        "worker_id": None,
    }
    await client.set(_task_key(task_id), record, ttl_seconds=ttl_seconds)
    await client.rpush(QUEUE_REASON_PENDING, task_id)
    return record


async def get_task(client, task_id: str) -> Optional[dict]:
    value, exists = await client.get(_task_key(task_id))
    return value if exists else None


async def list_queued_tasks(client) -> list[dict]:
    """LRANGE 队列 + 并发 GET 每个 task:{id}，构造待执行任务列表。

    并发 GET 是为了避免在队列较长（几十条）时被 N 次串行 RTT 拖慢
    /api/requestQueueStatus 的响应。
    """
    task_ids = await client.lrange(QUEUE_REASON_PENDING, 0, -1)
    if not task_ids:
        return []
    results = await asyncio.gather(*[get_task(client, tid) for tid in task_ids])
    return [r for r in results if r is not None]


async def list_running_tasks(client) -> list[dict]:
    """扫描 task:* 前缀 + 过滤 status=running。

    当前规模（最多 MAX_CONCURRENT_REASONING 个在跑 + 过去 24h 的历史）
    直接全扫没问题；如果将来量级上去需要单独维护一个 running 集合。
    """
    keys = await client.keys(prefix=TASK_KEY_PREFIX)
    if not keys:
        return []
    tasks = await asyncio.gather(*[
        get_task(client, k[len(TASK_KEY_PREFIX):]) for k in keys
    ])
    running = [t for t in tasks if t and t.get("status") == STATUS_RUNNING]
    running.sort(key=lambda x: x.get("start_time") or 0)
    return running


# ------------------------------------------------------------------ WorkerPool


class WorkerPool:
    """服务进程内置的 worker 池。

    每个 worker 独立循环：BLPOP(阻塞等) -> 读 task KV -> 标记 running
    -> 调 executor -> 写回 done/failed。worker 之间通过 redis_server 的
    BLPOP 队列协调，保证同一个 task_id 只会被一个 worker pop 到（BLPOP 的
    原生语义）。

    parameters
    ----------
    client : RedisServerClient
        已连接的异步客户端。pool 不负责 close 它，由调用方管理生命周期。
    executor : async callable(request_payload: dict) -> result_dict
        业务执行器。必须是 async 的；内部自行用 asyncio.to_thread 切同步工作到线程。
        抛异常会被 pool 捕获并标记 task failed，不会打爆 worker 循环。
    worker_count : int
        起多少个并发 worker。对应老版本的 MAX_CONCURRENT_REASONING。
    blpop_timeout_seconds : float
        单次 BLPOP 阻塞时长；到点没拿到任务就重新来一轮，
        这样 `stop()` 时 worker 最多卡 blpop_timeout 秒就能退出。
    """

    def __init__(
        self,
        client,
        executor: Callable[[dict], Awaitable[dict]],
        *,
        worker_count: int = 10,
        task_ttl_seconds: float = DEFAULT_TASK_TTL_SECONDS,
        queue_name: str = QUEUE_REASON_PENDING,
        blpop_timeout_seconds: float = 30.0,
    ) -> None:
        self.client = client
        self.executor = executor
        self.worker_count = worker_count
        self.task_ttl_seconds = task_ttl_seconds
        self.queue_name = queue_name
        self.blpop_timeout_seconds = blpop_timeout_seconds

        self._tasks: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------ 生命周期

    async def start(self) -> None:
        if self._tasks:
            return
        self._stop_event.clear()
        for i in range(self.worker_count):
            t = asyncio.create_task(self._worker_loop(i), name=f"reason-worker-{i}")
            self._tasks.append(t)
        logger.info(f"[WorkerPool] 已启动 {self.worker_count} 个 worker，queue={self.queue_name}")

    async def stop(self) -> None:
        if not self._tasks:
            return
        self._stop_event.set()
        # 先等 worker 自己通过 blpop 超时退出；实在不退的强行 cancel。
        await asyncio.sleep(0)
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()
        logger.info("[WorkerPool] 已停止全部 worker")

    # ------------------------------------------------------------- worker 主体

    async def _worker_loop(self, worker_id: int) -> None:
        logger.info(f"[WorkerPool] worker-{worker_id} 启动")
        while not self._stop_event.is_set():
            try:
                task_id, ok = await self.client.blpop(
                    self.queue_name,
                    timeout_seconds=self.blpop_timeout_seconds,
                )
                if not ok:
                    continue
                await self._run_one(worker_id, task_id)
            except asyncio.CancelledError:
                break
            except _TRANSIENT_NETWORK_ERRORS as e:
                # 网关 504 / 连接重置 / 请求超时等"到 redis_server 链路"的瞬时错误：
                # 典型场景是中间网关 proxy_read_timeout < blpop_timeout，
                # 后端本身没事，下一轮重试即可。降成 warning，避免刷屏。
                logger.warning(
                    f"[WorkerPool] worker-{worker_id} BLPOP 网络瞬时错误（通常是网关超时），"
                    f"0.5s 后重试: {e}"
                )
                try:
                    await asyncio.sleep(0.5)
                except asyncio.CancelledError:
                    break
            except Exception as e:
                # 其余异常当真故障处理：打完整堆栈 + 更长退避，避免陷入爆炸循环。
                logger.exception(f"[WorkerPool] worker-{worker_id} 循环异常: {e}")
                try:
                    await asyncio.sleep(1.0)
                except asyncio.CancelledError:
                    break
        logger.info(f"[WorkerPool] worker-{worker_id} 退出")

    async def _run_one(self, worker_id: int, task_id: Any) -> None:
        if not isinstance(task_id, str):
            # 理论上不可能——队列里只放 uuid 字符串。
            # 若出现说明上游不遵守协议往队列塞了别的东西，直接丢弃并告警。
            logger.error(f"[WorkerPool] 非字符串 task_id={task_id!r}，丢弃")
            return

        record = await get_task(self.client, task_id)
        if record is None:
            logger.warning(
                f"[WorkerPool] worker-{worker_id} 拉到 task_id={task_id}，"
                f"但 KV 里没有对应记录（可能已过期或被清空），跳过"
            )
            return

        record["status"] = STATUS_RUNNING
        record["start_time"] = _now()
        record["worker_id"] = worker_id
        await self.client.set(
            _task_key(task_id), record, ttl_seconds=self.task_ttl_seconds
        )

        try:
            result = await self.executor(record["request"])
            record["status"] = STATUS_DONE
            record["result"] = result
            record["error"] = None
        except Exception as e:
            logger.exception(f"[WorkerPool] worker-{worker_id} task={task_id} 执行失败")
            record["status"] = STATUS_FAILED
            record["result"] = None
            record["error"] = f"{type(e).__name__}: {e}"

        record["end_time"] = _now()
        await self.client.set(
            _task_key(task_id), record, ttl_seconds=self.task_ttl_seconds
        )

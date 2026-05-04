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
- 待执行队列 key : ``queue:reason:pending``，存入内容为 task_id 字符串；
- 运行中索引 key : ``queue:reason:running``，存入 worker 已拉起但未结束的 task_id；
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
QUEUE_REASON_RUNNING = "queue:reason:running"
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


async def _retry_short_request(
    coro_factory: Callable[[], Awaitable[Any]],
    *,
    attempts: int = 1,
    backoff_seconds: float = 0.3,
    op_name: str = "redis_server short request",
) -> Any:
    """对'短请求型'调用做最多 attempts+1 次尝试，瞬时网络错误退避后重试。

    背景：BLPOP 长轮询在客户端会偶发 RemoteProtocolError / PoolTimeout
    （网关回收 keepalive、链路抖动等）。submit/queue_status 等关键路径上的
    短请求若不重试，单次抖动就会冒成 5xx。这里提供一个轻量包装：
    捕获 ``_TRANSIENT_NETWORK_ERRORS``、退避一次再试，仍失败才上抛。

    parameters
    ----------
    coro_factory :
        每次尝试都重新调用一次以构造**新的** awaitable，避免把已 await 过的协程
        二次 await（asyncio 会报 RuntimeError: cannot reuse already awaited）。
    attempts :
        失败后再追加的重试次数。0 = 不重试（行为与裸调一致），1 = 最多调 2 次。
    backoff_seconds :
        每次失败后的固定退避；不做指数退避，避免提交路径感知到明显延迟。
    op_name :
        仅用于日志；标识哪一类操作触发了重试。
    """
    last_err: Optional[BaseException] = None
    for attempt in range(attempts + 1):
        try:
            return await coro_factory()
        except _TRANSIENT_NETWORK_ERRORS as e:
            last_err = e
            if attempt >= attempts:
                raise
            logger.warning(
                f"[{op_name}] 第 {attempt + 1} 次失败 ({type(e).__name__}: {e})，"
                f"{backoff_seconds:.2f}s 后重试"
            )
            await asyncio.sleep(backoff_seconds)
    # 理论不可达：上面的循环要么 return 要么 raise；写在这里只是给静态检查兜底。
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"[{op_name}] 重试逻辑异常退出")


async def submit_task(
    client,
    request_payload: dict,
    *,
    ttl_seconds: float = DEFAULT_TASK_TTL_SECONDS,
    retry_attempts: int = 1,
    retry_backoff_seconds: float = 0.3,
) -> dict:
    """生成 task_id、写入 task:{id}、RPUSH 到 queue:reason:pending。

    写 KV 再 push 队列的顺序不能颠倒：worker BLPOP 拿到 task_id 后会立刻
    GET task:{id}，若此时 KV 里还没有该条 record，会被 worker 记 warning 并
    直接丢弃（见 ``WorkerPool._run_one``）。

    set / rpush 这两步默认带 1 次轻量重试以吸收链路瞬抖（PoolTimeout、网关 504、
    连接重置等）。任意一步**最终**失败都会原样抛回路由层，由 /api/reason/submit
    返回 500；rpush 失败时会尽力清理已写入的 task:{id}，避免它在 KV 里成为孤儿
    （没人会去消费、占着 TTL 时长）。
    """
    task_id = str(uuid.uuid4())
    request_payload = dict(request_payload)
    request_payload["taskId"] = task_id
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
        # 进程实例标识：只有 worker 真正拉起这条任务时才会由 _run_one 写入当前进程的
        # INSTANCE_ID；pending 阶段保持为 None。sweeper / startup 清理据此区分
        # "本进程正在跑的 running" vs "上一代进程崩掉留下的僵尸 running"。
        "instance_id": None,
    }

    await _retry_short_request(
        lambda: client.set(_task_key(task_id), record, ttl_seconds=ttl_seconds),
        attempts=retry_attempts,
        backoff_seconds=retry_backoff_seconds,
        op_name=f"submit_task.set task={task_id}",
    )
    try:
        await _retry_short_request(
            lambda: client.rpush(QUEUE_REASON_PENDING, task_id),
            attempts=retry_attempts,
            backoff_seconds=retry_backoff_seconds,
            op_name=f"submit_task.rpush task={task_id}",
        )
    except Exception:
        # rpush 失败：尽力把刚写入的 task KV 删掉，避免遗留孤儿记录占 TTL。
        # 删除本身也可能因为同一波抖动失败，捕获后忽略——24h TTL 自动兜底。
        try:
            await client.delete(_task_key(task_id))
        except Exception as cleanup_err:
            logger.warning(
                f"[submit_task] rpush 失败后清理 task={task_id} 也失败，"
                f"将依赖 TTL 过期: {cleanup_err}"
            )
        raise
    return record


async def get_task(client, task_id: str) -> Optional[dict]:
    value, exists = await client.get(_task_key(task_id))
    return value if exists else None


async def _add_running_task(client, task_id: str) -> None:
    """把 task_id 加入 running 索引；失败只记 warning，不中断实际执行。"""
    try:
        await client.rpush(QUEUE_REASON_RUNNING, task_id)
    except Exception as e:
        logger.warning(f"[running_index] task={task_id} 加入 running 索引失败: {e}")


async def _remove_running_task(client, task_id: str) -> None:
    """从 running 索引删除 task_id 的全部残留项；失败只记 warning。"""
    try:
        await client.lrem(QUEUE_REASON_RUNNING, task_id, count=0)
    except Exception as e:
        logger.warning(f"[running_index] task={task_id} 移除 running 索引失败: {e}")


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


async def clean_queued_tasks(client) -> list[dict]:
    """清除仍停留在 pending 队列里的任务。

    只以 redis list 当前快照为候选；每个 task_id 先 LREM，只有实际从队列移除成功
    才删除对应 task:{id} 记录。若扫描期间 task 已被 worker BLPOP 取走，LREM 会返回 0,
    本函数会跳过删除 KV，避免误删已经 running 的任务记录。
    """
    task_ids = await client.lrange(QUEUE_REASON_PENDING, 0, -1)
    if not task_ids:
        return []

    cleaned: list[dict] = []
    for task_id in task_ids:
        entry: dict = {
            "task_id": str(task_id),
            "removed_from_queue": 0,
            "deleted_task": False,
        }
        if not isinstance(task_id, str):
            entry["skipped_reason"] = "non-string task_id"
            cleaned.append(entry)
            continue

        try:
            removed = await client.lrem(QUEUE_REASON_PENDING, task_id, count=0)
            entry["removed_from_queue"] = removed
            if removed <= 0:
                entry["skipped_reason"] = "not found in queue during cleanup"
                cleaned.append(entry)
                continue

            entry["deleted_task"] = await client.delete(_task_key(task_id))
        except Exception as e:
            entry["skipped_reason"] = f"cleanup failed: {e}"
            logger.warning(f"[clean_queued_tasks] task={task_id} 清理异常: {e}")
        cleaned.append(entry)

    total_removed = sum(int(e.get("removed_from_queue") or 0) for e in cleaned)
    if total_removed:
        logger.info(
            f"[clean_queued_tasks] 从 {QUEUE_REASON_PENDING} 清除 {total_removed} 条队列项"
        )
    return cleaned


async def list_running_tasks(client) -> list[dict]:
    """读取 running 索引 + GET 少量候选记录。

    旧实现会 ``keys("task:")`` 扫所有历史 task，再逐条 GET 后过滤 running。
    由于 task TTL 默认 7 天，历史任务一多，/api/requestQueueStatus 会越来越慢。
    现在改为只读 ``queue:reason:running``，规模理论上不超过 worker_count；
    若索引里残留已完成/过期任务，本函数会顺手 LREM 自愈。
    """
    task_ids = await client.lrange(QUEUE_REASON_RUNNING, 0, -1)
    if not task_ids:
        return []

    # list 不是 set，极端情况下可能有重复残留；保持顺序去重，避免重复 GET。
    seen: set[str] = set()
    unique_task_ids: list[str] = []
    stale_task_ids: list[str] = []
    for task_id in task_ids:
        if not isinstance(task_id, str):
            continue
        if task_id in seen:
            stale_task_ids.append(task_id)
            continue
        seen.add(task_id)
        unique_task_ids.append(task_id)

    records = await asyncio.gather(*[get_task(client, tid) for tid in unique_task_ids])
    running: list[dict] = []
    for task_id, record in zip(unique_task_ids, records):
        if record and record.get("status") == STATUS_RUNNING:
            running.append(record)
        else:
            stale_task_ids.append(task_id)

    if stale_task_ids:
        await asyncio.gather(
            *[_remove_running_task(client, tid) for tid in stale_task_ids],
            return_exceptions=True,
        )

    running.sort(key=lambda x: x.get("start_time") or 0)
    return running


async def cleanup_stale_running_tasks(
    client,
    current_instance_id: str,
    *,
    ttl_seconds: float = DEFAULT_TASK_TTL_SECONDS,
    error_message: str = "server restarted while running",
) -> int:
    """启动时 / 恢复时清理"上一代进程留下的僵尸 running 记录"。

    判据：``status == running`` 且 ``instance_id != current_instance_id``
    （包含字段缺失 / 为 None 的老记录——那些一定是改造前写入的，不可能属于本进程）。
    命中的记录就地改写为 ``status=failed`` + ``error=<error_message>`` + ``end_time=now``,
    供客户端轮询立即感知。

    典型调用点：``app.py`` 的 lifespan startup，在 WorkerPool.start() 之前执行一次。

    返回被清理的记录数量，便于上层打 info 日志。
    """
    keys = await client.keys(prefix=TASK_KEY_PREFIX)
    if not keys:
        return 0

    records = await asyncio.gather(*[
        get_task(client, k[len(TASK_KEY_PREFIX):]) for k in keys
    ])

    cleaned = 0
    for record in records:
        if not record:
            continue
        if record.get("status") != STATUS_RUNNING:
            continue
        if record.get("instance_id") == current_instance_id:
            # 本进程自己正在跑的，不碰。
            continue

        task_id = record.get("task_id")
        if not task_id:
            continue

        record["status"] = STATUS_FAILED
        record["result"] = None
        record["error"] = error_message
        record["end_time"] = _now()
        try:
            await client.set(_task_key(task_id), record, ttl_seconds=ttl_seconds)
            await _remove_running_task(client, task_id)
            cleaned += 1
        except Exception as e:
            # 单条清理失败不影响其它条，只记 warning 继续扫。
            logger.warning(
                f"[cleanup_stale_running_tasks] task={task_id} 写回 failed 异常: {e}"
            )

    if cleaned:
        logger.info(
            f"[cleanup_stale_running_tasks] 清理 {cleaned} 条僵尸 running 记录 "
            f"(instance_id={current_instance_id})"
        )
    return cleaned


async def cleanup_running_tasks_by_age(
    client,
    threshold_seconds: float,
    *,
    current_instance_id: str = "",
    dry_run: bool = False,
    ttl_seconds: float = DEFAULT_TASK_TTL_SECONDS,
    error_message: Optional[str] = None,
) -> list[dict]:
    """按 running 耗时阈值清理任务：命中项改写为 failed。

    与 ``cleanup_stale_running_tasks``（按 instance_id 匹配，专治"重启后的僵尸"）不同,
    本函数的判据是"now - start_time > threshold_seconds"，**不区分** instance_id,
    用于服务**未重启**、某些 task 因下游/逻辑卡死长时间停在 running 的运维场景。

    parameters
    ----------
    threshold_seconds :
        running_seconds（now - start_time）超过该阈值的 task 才会被视为超时。
        负数会被当成 0（等价全部 running 都命中）。
    current_instance_id :
        可选。仅用于在返回 entry 里标注 ``matches_current_instance``，方便调用方
        识别"这条可能是本进程真实在跑的"而谨慎处理；本函数**不会**因此跳过清理。
    dry_run :
        True 时只返回命中候选、不写回 redis，便于运维先预览再决定是否真清。
    error_message :
        写入被清理任务的 ``error`` 字段；默认带阈值信息，便于事后排查。
    ttl_seconds :
        写回时继续延续的 TTL。

    返回
    ----
    list[dict]，按 running_seconds 降序（最可疑的在前），每条包含：
      task_id / policy_id / question / running_seconds / start_time /
      worker_id / instance_id / matches_current_instance / cleaned /
      skipped_reason(可选)。

    并发安全
    --------
    写回 failed 前做一次二次 GET，若期间 worker 正好写了 done/failed，则本次跳过,
    缩窄"把已完成任务回写成 failed"的 race 窗口。
    """
    if threshold_seconds < 0:
        threshold_seconds = 0.0
    if error_message is None:
        error_message = f"manually cleaned: running over {threshold_seconds:.0f}s"

    keys = await client.keys(prefix=TASK_KEY_PREFIX)
    if not keys:
        return []

    records = await asyncio.gather(*[
        get_task(client, k[len(TASK_KEY_PREFIX):]) for k in keys
    ])

    now = _now()
    candidates: list[tuple[dict, float]] = []
    for record in records:
        if not record:
            continue
        if record.get("status") != STATUS_RUNNING:
            continue
        start = record.get("start_time")
        if start is None:
            continue
        try:
            age = now - float(start)
        except (TypeError, ValueError):
            continue
        if age < threshold_seconds:
            continue
        candidates.append((record, age))

    # 最可疑（跑得最久）的排前面，便于调用方直观看到。
    candidates.sort(key=lambda x: x[1], reverse=True)

    result: list[dict] = []
    for record, age in candidates:
        task_id = record.get("task_id")
        request_payload = record.get("request") or {}
        entry: dict = {
            "task_id": task_id or "",
            "policy_id": request_payload.get("policyId", ""),
            "question": request_payload.get("question", ""),
            "running_seconds": round(age, 3),
            "start_time": record.get("start_time"),
            "worker_id": record.get("worker_id"),
            "instance_id": record.get("instance_id"),
            "matches_current_instance": bool(
                current_instance_id
                and record.get("instance_id") == current_instance_id
            ),
            "cleaned": False,
        }

        if dry_run or not task_id:
            if not task_id:
                entry["skipped_reason"] = "missing task_id"
            result.append(entry)
            continue

        # 二次校验，减少与 worker 正常完成之间的 race。
        latest = await get_task(client, task_id)
        if not latest or latest.get("status") != STATUS_RUNNING:
            entry["skipped_reason"] = "status changed during scan"
            result.append(entry)
            continue

        latest["status"] = STATUS_FAILED
        latest["result"] = None
        latest["error"] = error_message
        latest["end_time"] = _now()
        try:
            await client.set(
                _task_key(task_id), latest, ttl_seconds=ttl_seconds
            )
            await _remove_running_task(client, task_id)
            entry["cleaned"] = True
        except Exception as e:
            entry["skipped_reason"] = f"write back failed: {e}"
            logger.warning(
                f"[cleanup_running_tasks_by_age] task={task_id} 写回 failed 异常: {e}"
            )
        result.append(entry)

    if not dry_run:
        cleaned = sum(1 for e in result if e.get("cleaned"))
        if cleaned:
            logger.info(
                f"[cleanup_running_tasks_by_age] 清理 {cleaned}/{len(result)} 条超时 running 任务 "
                f"(threshold={threshold_seconds}s)"
            )
    return result


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
        单次 BLPOP 阻塞时长；到点没拿到任务就重新来一轮,
        这样 `stop()` 时 worker 最多卡 blpop_timeout 秒就能退出。
    instance_id : str
        当前服务进程的实例 ID（通常 `lifespan` 里生成一个 uuid 注入）。
        worker 把任务标记为 running 时会把它写入 `task:{id}.instance_id`，
        供 sweeper / startup 清理识别"本进程的 running" vs "上一代的僵尸"。
    executor_timeout_seconds : float
        单次 executor 调用的硬超时（秒）。超过即 asyncio.wait_for 抛 TimeoutError,
        worker 协程跳出、槽位释放、task 被标为 failed。
        默认 0 / 负数 / None = 不套超时（保留老行为，仅用于向后兼容）。
        注意：wait_for 只能中断 awaitable 链；若 executor 内部通过 asyncio.to_thread
        跑同步代码，超时后 OS 线程不会停，依赖下游 httpx/LLM client 自身的 read timeout
        来防止线程泄漏。
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
        instance_id: str = "",
        executor_timeout_seconds: Optional[float] = None,
    ) -> None:
        self.client = client
        self.executor = executor
        self.worker_count = worker_count
        self.task_ttl_seconds = task_ttl_seconds
        self.queue_name = queue_name
        self.blpop_timeout_seconds = blpop_timeout_seconds
        self.instance_id = instance_id
        self.executor_timeout_seconds = executor_timeout_seconds

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
                # 显式带上异常类型：很多底层异常（RemoteProtocolError/ReadError 等）
                # 构造时不带 message，只打 {e} 会变空字符串，无法定位根因。
                logger.warning(
                    f"[WorkerPool] worker-{worker_id} BLPOP 网络瞬时错误（通常是网关超时），"
                    f"0.5s 后重试: {type(e).__name__}: {e}"
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
        record["instance_id"] = self.instance_id
        await self.client.set(
            _task_key(task_id), record, ttl_seconds=self.task_ttl_seconds
        )
        await _add_running_task(self.client, task_id)

        timeout = self.executor_timeout_seconds
        try:
            if timeout is not None and timeout > 0:
                result = await asyncio.wait_for(
                    self.executor(record["request"]),
                    timeout=timeout,
                )
            else:
                result = await self.executor(record["request"])
            record["status"] = STATUS_DONE
            record["result"] = result
            record["error"] = None
        except asyncio.TimeoutError:
            # 硬超时：wait_for 会取消外层 awaitable 链，worker 协程借此跳出、槽位释放。
            # 但若 executor 底下是 asyncio.to_thread 跑的同步代码（当前 _reason_executor 就是）,
            # 那个 OS 线程不会被取消，会继续占资源直到下游自己返回，依赖下游 read timeout 兜底。
            logger.error(
                f"[WorkerPool] worker-{worker_id} task={task_id} executor 超过硬超时 {timeout}s，标记 failed"
            )
            record["status"] = STATUS_FAILED
            record["result"] = None
            record["error"] = f"executor timeout after {timeout}s"
        except Exception as e:
            logger.exception(f"[WorkerPool] worker-{worker_id} task={task_id} 执行失败")
            record["status"] = STATUS_FAILED
            record["result"] = None
            record["error"] = f"{type(e).__name__}: {e}"

        record["end_time"] = _now()
        await self.client.set(
            _task_key(task_id), record, ttl_seconds=self.task_ttl_seconds
        )
        await _remove_running_task(self.client, task_id)

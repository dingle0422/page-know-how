"""内存存储引擎：KV（带 TTL） + List（FIFO 队列，支持阻塞 pop） + JSON 快照持久化。

设计要点
--------
1. 单进程 asyncio + 一把全局 `asyncio.Lock` 串行化所有写操作，
   避免引入 threading / 多把锁带来的复杂性。读路径同样走锁以保证 TTL 惰性清理原子性。
2. TTL 采用「惰性清理 + 后台周期扫描」：
   - 惰性：每次 get / exists 判断过期则就地删除；
   - 周期：后台任务每 `sweep_interval_seconds` 扫一遍，防止只写不读的冷 key 堆积。
3. 阻塞 pop（BLPOP 语义）：
   - 队列空时把 future 挂到 `_waiters[queue_name]`；
   - 下一次 rpush 若有 waiter 直接交付（不入队），O(1) 唤醒；
   - 超时清理时检查 fut 是否已被 set_result，若是则把值"回塞"队头，
     保证消息不会因为超时竞态丢失。
4. 持久化：JSON 快照（RDB 风格），启动加载 + 后台周期 dump + 退出 dump。
   写文件用「写 tmp → 原子 rename」避免宕机半写。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class _Entry:
    """KV 条目。expires_at 为 None 表示永不过期。"""

    value: Any
    expires_at: Optional[float] = None

    def is_expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at


class Store:
    """内存存储引擎，所有方法均为 coroutine。"""

    def __init__(
        self,
        snapshot_path: Optional[Path] = None,
        snapshot_interval_seconds: float = 30.0,
        sweep_interval_seconds: float = 60.0,
    ) -> None:
        self._kv: dict[str, _Entry] = {}
        self._queues: dict[str, deque] = {}
        self._waiters: dict[str, list[asyncio.Future]] = {}
        self._lock = asyncio.Lock()

        self.snapshot_path = snapshot_path
        self.snapshot_interval_seconds = snapshot_interval_seconds
        self.sweep_interval_seconds = sweep_interval_seconds

        self._bg_tasks: list[asyncio.Task] = []
        self._closed = False

    # ---------------------------------------------------------------- KV API

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        async with self._lock:
            expires_at = time.time() + ttl_seconds if ttl_seconds else None
            self._kv[key] = _Entry(value, expires_at)

    async def get(self, key: str) -> tuple[Any, bool]:
        """返回 (value, exists)。不存在或已过期均返回 (None, False)。"""
        async with self._lock:
            entry = self._kv.get(key)
            if entry is None:
                return None, False
            if entry.is_expired(time.time()):
                del self._kv[key]
                return None, False
            return entry.value, True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            return self._kv.pop(key, None) is not None

    async def exists(self, key: str) -> bool:
        async with self._lock:
            entry = self._kv.get(key)
            if entry is None:
                return False
            if entry.is_expired(time.time()):
                del self._kv[key]
                return False
            return True

    async def expire(self, key: str, ttl_seconds: float) -> bool:
        async with self._lock:
            entry = self._kv.get(key)
            if entry is None:
                return False
            entry.expires_at = time.time() + ttl_seconds
            return True

    async def ttl(self, key: str) -> Optional[float]:
        """返回剩余秒数；不存在返回 None；永不过期返回 -1。"""
        async with self._lock:
            entry = self._kv.get(key)
            if entry is None:
                return None
            if entry.expires_at is None:
                return -1.0
            remaining = entry.expires_at - time.time()
            if remaining <= 0:
                del self._kv[key]
                return None
            return remaining

    async def keys(self, prefix: str = "") -> list[str]:
        """返回所有 key（可按前缀过滤，便于调试/运维）。"""
        async with self._lock:
            now = time.time()
            result = []
            expired = []
            for k, e in self._kv.items():
                if e.is_expired(now):
                    expired.append(k)
                    continue
                if not prefix or k.startswith(prefix):
                    result.append(k)
            for k in expired:
                del self._kv[k]
            return result

    # ------------------------------------------------------------- Queue API

    async def rpush(self, name: str, value: Any) -> int:
        """尾入队，返回入队后队列长度。若有 BLPOP 等待者则直接交付。"""
        async with self._lock:
            waiters = self._waiters.get(name)
            while waiters:
                fut = waiters.pop(0)
                if not fut.done():
                    fut.set_result(value)
                    q = self._queues.get(name)
                    return len(q) if q else 0
            q = self._queues.setdefault(name, deque())
            q.append(value)
            return len(q)

    async def lpop(self, name: str) -> tuple[Any, bool]:
        """头出队（非阻塞），返回 (value, ok)。"""
        async with self._lock:
            q = self._queues.get(name)
            if not q:
                return None, False
            value = q.popleft()
            if not q:
                del self._queues[name]
            return value, True

    async def blpop(
        self,
        name: str,
        timeout_seconds: float,
    ) -> tuple[Any, bool]:
        """阻塞头出队。timeout_seconds<=0 表示无限等待。"""
        value, ok = await self.lpop(name)
        if ok:
            return value, True

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        async with self._lock:
            self._waiters.setdefault(name, []).append(fut)

        try:
            if timeout_seconds and timeout_seconds > 0:
                value = await asyncio.wait_for(fut, timeout=timeout_seconds)
            else:
                value = await fut
            return value, True
        except asyncio.TimeoutError:
            async with self._lock:
                waiters = self._waiters.get(name)
                if waiters and fut in waiters:
                    waiters.remove(fut)
                # 竞态：超时触发与 set_result 擦肩而过，把值补回队头
                if fut.done() and not fut.cancelled():
                    try:
                        recovered = fut.result()
                    except Exception:
                        recovered = None
                    else:
                        q = self._queues.setdefault(name, deque())
                        q.appendleft(recovered)
            return None, False

    async def llen(self, name: str) -> int:
        async with self._lock:
            q = self._queues.get(name)
            return len(q) if q else 0

    async def lrange(self, name: str, start: int, stop: int) -> list[Any]:
        """Redis LRANGE 语义：闭区间、支持负索引（stop=-1 表尾）。"""
        async with self._lock:
            q = self._queues.get(name)
            if not q:
                return []
            items = list(q)
            n = len(items)
            if start < 0:
                start = max(0, n + start)
            if stop < 0:
                stop = n + stop
            stop = min(stop, n - 1)
            if start > stop:
                return []
            return items[start : stop + 1]

    async def lrem(self, name: str, value: Any, count: int) -> int:
        """count>0 从头删 count 个；count<0 从尾删 |count| 个；count=0 删全部。"""
        async with self._lock:
            q = self._queues.get(name)
            if not q:
                return 0
            items = list(q)
            removed = 0
            if count > 0:
                out = []
                for x in items:
                    if removed < count and x == value:
                        removed += 1
                        continue
                    out.append(x)
            elif count < 0:
                reversed_out: list[Any] = []
                for x in reversed(items):
                    if removed < -count and x == value:
                        removed += 1
                        continue
                    reversed_out.append(x)
                reversed_out.reverse()
                out = reversed_out
            else:
                out = []
                for x in items:
                    if x == value:
                        removed += 1
                        continue
                    out.append(x)
            if out:
                self._queues[name] = deque(out)
            else:
                self._queues.pop(name, None)
            return removed

    # -------------------------------------------------------- Introspection

    async def stats(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "kv_count": len(self._kv),
                "queues": {k: len(v) for k, v in self._queues.items()},
                "blocking_waiters": {
                    k: len([w for w in v if not w.done()])
                    for k, v in self._waiters.items()
                    if v
                },
            }

    # --------------------------------------------------------- Housekeeping

    async def sweep_expired(self) -> int:
        async with self._lock:
            now = time.time()
            expired = [k for k, e in self._kv.items() if e.is_expired(now)]
            for k in expired:
                del self._kv[k]
            return len(expired)

    async def flushall(self) -> None:
        async with self._lock:
            self._kv.clear()
            self._queues.clear()
            # 唤醒所有阻塞等待者，避免永久挂起
            for waiters in self._waiters.values():
                for fut in waiters:
                    if not fut.done():
                        fut.cancel()
            self._waiters.clear()

    # --------------------------------------------------------- Snapshot I/O

    async def snapshot_to_dict(self) -> dict[str, Any]:
        async with self._lock:
            now = time.time()
            live_kv: dict[str, dict] = {}
            for k, e in self._kv.items():
                if e.is_expired(now):
                    continue
                live_kv[k] = {"value": e.value, "expires_at": e.expires_at}
            return {
                "version": 1,
                "snapshot_time": now,
                "kv": live_kv,
                "queues": {k: list(v) for k, v in self._queues.items() if v},
            }

    async def load_from_dict(self, data: dict[str, Any]) -> None:
        async with self._lock:
            self._kv.clear()
            self._queues.clear()
            for k, rec in data.get("kv", {}).items():
                self._kv[k] = _Entry(rec.get("value"), rec.get("expires_at"))
            for k, items in data.get("queues", {}).items():
                if items:
                    self._queues[k] = deque(items)

    async def save_snapshot(self) -> Optional[Path]:
        if self.snapshot_path is None:
            return None
        data = await self.snapshot_to_dict()
        path = self.snapshot_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        # 写 tmp → fsync → 原子替换，避免宕机半写
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
        return path

    async def load_snapshot(self) -> bool:
        if self.snapshot_path is None or not self.snapshot_path.exists():
            return False
        try:
            with open(self.snapshot_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.exception(f"加载快照失败，将从空状态启动: {e}")
            return False
        await self.load_from_dict(data)
        kv_n = len(data.get("kv", {}))
        q_total = sum(len(v) for v in data.get("queues", {}).values())
        logger.info(f"快照已加载: kv={kv_n}, queue_items={q_total}, path={self.snapshot_path}")
        return True

    # ----------------------------------------------- Background bookkeeping

    async def start_background_tasks(self) -> None:
        """启动周期快照 + 过期扫描任务。由 server lifespan 调用。"""
        if self._bg_tasks:
            return
        if self.snapshot_path is not None and self.snapshot_interval_seconds > 0:
            self._bg_tasks.append(asyncio.create_task(self._snapshot_loop()))
        if self.sweep_interval_seconds > 0:
            self._bg_tasks.append(asyncio.create_task(self._sweep_loop()))

    async def stop_background_tasks(self) -> None:
        self._closed = True
        for t in self._bg_tasks:
            t.cancel()
        for t in self._bg_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._bg_tasks.clear()

    async def _snapshot_loop(self) -> None:
        while not self._closed:
            try:
                await asyncio.sleep(self.snapshot_interval_seconds)
                path = await self.save_snapshot()
                if path:
                    logger.debug(f"周期快照落盘: {path}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"周期快照失败: {e}")

    async def _sweep_loop(self) -> None:
        while not self._closed:
            try:
                await asyncio.sleep(self.sweep_interval_seconds)
                n = await self.sweep_expired()
                if n:
                    logger.debug(f"过期扫描清理 {n} 条 KV")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"过期扫描失败: {e}")

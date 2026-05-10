"""inference 阶段的 redis 状态读写门面。

设计要点：

- 单 key：``inference:task:{taskId}`` 存一份 JSON 快照（schema 见 :func:`make_initial_snapshot`）。
- 写入路径统一 ``get -> mutate -> recompute_aggregates -> set``，
  在进程内用 per-task ``asyncio.Lock`` 串行化，避免 preview/skills/react 互相覆盖。
- 接口侧 SSE 直接消费 ``snapshot["think"]`` / ``snapshot["answer"]`` 两个聚合字段，
  这两个字段每次写入时都由 :func:`recompute_aggregates` 根据
  ``REACT_INTERMEDIATE_THINK_ENABLED`` 开关重新计算，实现端不需要再做拼接。

注意：底层 ``RedisServerClient.set`` 会整包覆盖 value，所以**所有写入端必须走本模块的
``update`` 包装**，否则会丢字段。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Awaitable, Callable, Optional

from redis_server.client import RedisServerClient

from . import config

Snapshot = dict[str, Any]
Mutator = Callable[[Snapshot], Optional[Snapshot]]


# ---------------------------------------------------------------- 快照 schema

def make_initial_snapshot(
    task_id: str,
    question: str,
    policy_id: str,
    *,
    intermediate_think_enabled: Optional[bool] = None,
) -> Snapshot:
    """构造空的初始快照。

    ``intermediate_think_enabled`` 显式存进快照，避免 worker 进程与
    SSE 转发进程对开关取值不一致时聚合规则跑偏。
    """

    now = time.time()
    flag = (
        bool(intermediate_think_enabled)
        if intermediate_think_enabled is not None
        else bool(config.REACT_INTERMEDIATE_THINK_ENABLED)
    )
    return {
        "taskId": task_id,
        "policyId": policy_id,
        "question": question,
        "status": "pending",
        "preview": {"think": "", "answer": "", "done": False},
        "skills": [],
        "react": {"round": 0, "chunks": []},
        "intermediateThinkEnabled": flag,
        "think": "",
        "answer": "",
        "error": None,
        "createdAt": now,
        "updatedAt": now,
    }


# ---------------------------------------------------------------- 聚合规则

def recompute_aggregates(
    snapshot: Snapshot,
    intermediate_think_enabled: Optional[bool] = None,
) -> tuple[str, str]:
    """根据 react/preview 子结构重算接口 ``think`` / ``answer``。

    - 默认模式（开关关）：``preview.think + preview.answer + chunks[final-1].answer + chunks[final].think``
    - 开关开：``preview.* + 非最终轮(think+answer) 全拼接 + chunks[final].think``
    - ``answer`` 始终只取最终轮 ``chunk.answer``，未到最终轮则为空字符串。
    """

    if intermediate_think_enabled is None:
        intermediate_think_enabled = bool(
            snapshot.get("intermediateThinkEnabled", config.REACT_INTERMEDIATE_THINK_ENABLED)
        )

    preview = snapshot.get("preview") or {}
    react = snapshot.get("react") or {}
    chunks = react.get("chunks") or []

    parts: list[str] = []
    p_think = (preview.get("think") or "").strip()
    p_answer = (preview.get("answer") or "").strip()
    if p_think:
        parts.append(p_think)
    if p_answer:
        parts.append(p_answer)

    answer = ""
    if chunks:
        final_idx = len(chunks) - 1
        final_chunk = chunks[final_idx] or {}
        final_think = (final_chunk.get("think") or "").strip()
        final_answer = (final_chunk.get("answer") or "").strip()

        if intermediate_think_enabled:
            for i in range(final_idx):
                c = chunks[i] or {}
                c_think = (c.get("think") or "").strip()
                c_answer = (c.get("answer") or "").strip()
                if c_think:
                    parts.append(c_think)
                if c_answer:
                    parts.append(c_answer)
        else:
            prev_idx = final_idx - 1
            if prev_idx >= 0:
                prev_answer = ((chunks[prev_idx] or {}).get("answer") or "").strip()
                if prev_answer:
                    parts.append(prev_answer)

        if final_think:
            parts.append(final_think)
        answer = final_answer

    think = "\n".join(parts).strip()
    return think, answer


# ---------------------------------------------------------------- 主门面


class RedisStream:
    """围绕 ``RedisServerClient`` 的薄封装，所有写都走 :meth:`update`。

    ``RedisServerClient`` 由调用方注入（通常复用 ``app.py`` lifespan 里的 ``_redis_client``），
    本类不持有连接生命周期，只负责语义层的快照读写与并发串行化。
    """

    def __init__(
        self,
        client: RedisServerClient,
        *,
        key_prefix: str = config.TASK_KEY_PREFIX,
        ttl_seconds: float = config.TASK_TTL_SECONDS,
    ) -> None:
        self._client = client
        self._key_prefix = key_prefix
        self._ttl = ttl_seconds
        # per-task asyncio.Lock：必须在事件循环里第一次访问时再创建，
        # 这样避免在模块导入时就绑定到错误的 loop 上。
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    # ------------------------------------------------------------ helpers

    def key_for(self, task_id: str) -> str:
        return f"{self._key_prefix}{task_id}"

    async def _lock_for(self, task_id: str) -> asyncio.Lock:
        # 双重检查：fast-path 不抢全局锁，慢 path 才落到 guard 里 setdefault。
        lock = self._locks.get(task_id)
        if lock is not None:
            return lock
        async with self._locks_guard:
            lock = self._locks.get(task_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[task_id] = lock
            return lock

    # ------------------------------------------------------------ basics

    async def exists(self, task_id: str) -> bool:
        return await self._client.exists(self.key_for(task_id))

    async def get(self, task_id: str) -> Optional[Snapshot]:
        value, ok = await self._client.get(self.key_for(task_id))
        if not ok or value is None:
            return None
        if not isinstance(value, dict):
            # 兜底：服务端理论上应该原样返回 dict；遇到坏数据就当成不存在处理。
            return None
        return value

    async def init(
        self,
        question: str,
        policy_id: str,
        *,
        task_id: Optional[str] = None,
        intermediate_think_enabled: Optional[bool] = None,
        overwrite: bool = False,
    ) -> tuple[str, Snapshot]:
        """如果 task 不存在则创建并返回快照；存在时按 ``overwrite`` 决定是否覆盖。"""

        tid = task_id or str(uuid.uuid4())
        lock = await self._lock_for(tid)
        async with lock:
            if not overwrite:
                current = await self.get(tid)
                if current is not None:
                    return tid, current
            snapshot = make_initial_snapshot(
                tid,
                question,
                policy_id,
                intermediate_think_enabled=intermediate_think_enabled,
            )
            await self._client.set(self.key_for(tid), snapshot, ttl_seconds=self._ttl)
            return tid, snapshot

    # ------------------------------------------------------------ writes

    async def update(self, task_id: str, mutator: Mutator) -> Snapshot:
        """``get -> mutate -> recompute -> set`` 单事务（进程内锁）。

        ``mutator`` 可以原地改也可以返回新 dict（返回 ``None`` 视为原地改）。
        ``think`` / ``answer`` / ``updatedAt`` 由本方法统一维护，mutator 不需要管。
        """

        lock = await self._lock_for(task_id)
        async with lock:
            current = await self.get(task_id)
            if current is None:
                # 兜底创建一个最小快照，避免 race 时丢数据。
                current = make_initial_snapshot(task_id, "", "")
            mutated = mutator(current)
            snapshot = mutated if isinstance(mutated, dict) else current
            think, answer = recompute_aggregates(snapshot)
            snapshot["think"] = think
            snapshot["answer"] = answer
            snapshot["updatedAt"] = time.time()
            await self._client.set(self.key_for(task_id), snapshot, ttl_seconds=self._ttl)
            return snapshot

    async def set_status(
        self,
        task_id: str,
        status: str,
        *,
        error: Optional[str] = None,
    ) -> Snapshot:
        def _mut(s: Snapshot) -> None:
            s["status"] = status
            if error is not None:
                s["error"] = error

        return await self.update(task_id, _mut)

    # ----- preview ---------------------------------------------------

    async def append_preview(
        self,
        task_id: str,
        channel: str,
        delta: str,
    ) -> Snapshot:
        """把 LLM 流式增量追加到 preview.think / preview.answer。"""

        if channel not in {"think", "answer"}:
            raise ValueError(f"preview channel 必须是 think|answer, got {channel!r}")
        if not delta:
            current = await self.get(task_id)
            if current is None:
                current = make_initial_snapshot(task_id, "", "")
            return current

        def _mut(s: Snapshot) -> None:
            preview = s.setdefault(
                "preview", {"think": "", "answer": "", "done": False}
            )
            preview[channel] = (preview.get(channel) or "") + delta

        return await self.update(task_id, _mut)

    async def set_preview_done(self, task_id: str, done: bool = True) -> Snapshot:
        def _mut(s: Snapshot) -> None:
            preview = s.setdefault(
                "preview", {"think": "", "answer": "", "done": False}
            )
            preview["done"] = bool(done)

        return await self.update(task_id, _mut)

    # ----- skills ----------------------------------------------------

    async def set_skills(self, task_id: str, skills: list[dict]) -> Snapshot:
        def _mut(s: Snapshot) -> None:
            s["skills"] = list(skills or [])

        return await self.update(task_id, _mut)

    # ----- react -----------------------------------------------------

    async def ensure_react_chunk(self, task_id: str, round_idx: int) -> Snapshot:
        """确保 ``react.chunks[round_idx]`` 存在；若缺失则填空 chunk。"""

        def _mut(s: Snapshot) -> None:
            react = s.setdefault("react", {"round": 0, "chunks": []})
            chunks: list[dict] = react.setdefault("chunks", [])
            while len(chunks) <= round_idx:
                chunks.append({"think": "", "answer": "", "complete": False})
            react["round"] = max(int(react.get("round") or 0), round_idx)

        return await self.update(task_id, _mut)

    async def append_react_chunk_delta(
        self,
        task_id: str,
        round_idx: int,
        channel: str,
        delta: str,
        *,
        is_last_chunk: bool,
    ) -> Snapshot:
        """ReAct 单轮流式增量写入 ``react.chunks[round_idx]``。

        ``is_last_chunk`` 在这里只起"占位语义提示"作用——具体非最终轮的 think 是否
        进入接口 ``think`` 字段，已经由 :func:`recompute_aggregates` 按开关统一处理。
        本方法保证：无论开关如何，所有进来的 delta 都会被忠实写到对应 chunk，
        从而让聚合规则成为唯一真相。
        """

        if channel not in {"think", "answer"}:
            raise ValueError(f"react channel 必须是 think|answer, got {channel!r}")
        if not delta:
            current = await self.get(task_id)
            if current is None:
                current = make_initial_snapshot(task_id, "", "")
            return current

        def _mut(s: Snapshot) -> None:
            react = s.setdefault("react", {"round": 0, "chunks": []})
            chunks: list[dict] = react.setdefault("chunks", [])
            while len(chunks) <= round_idx:
                chunks.append({"think": "", "answer": "", "complete": False})
            chunks[round_idx][channel] = (chunks[round_idx].get(channel) or "") + delta
            react["round"] = max(int(react.get("round") or 0), round_idx)
            # 标记仅供下游观测，is_last_chunk 影响聚合（通过 chunks 的最后一项判断）。
            if is_last_chunk:
                chunks[round_idx]["isFinal"] = True

        return await self.update(task_id, _mut)

    async def set_react_chunk_complete(
        self,
        task_id: str,
        round_idx: int,
        *,
        complete: bool = True,
        verdict: Optional[str] = None,
    ) -> Snapshot:
        def _mut(s: Snapshot) -> None:
            react = s.setdefault("react", {"round": 0, "chunks": []})
            chunks: list[dict] = react.setdefault("chunks", [])
            while len(chunks) <= round_idx:
                chunks.append({"think": "", "answer": "", "complete": False})
            chunks[round_idx]["complete"] = bool(complete)
            if verdict is not None:
                chunks[round_idx]["verdict"] = verdict

        return await self.update(task_id, _mut)


# ---------------------------------------------------------------- 便捷函数

async def with_recompute(
    stream: RedisStream,
    task_id: str,
    work: Callable[[Snapshot], Awaitable[Optional[Snapshot]]],
) -> Snapshot:
    """``update`` 的异步 mutator 版本，少数场景下需要在 mutator 内做 await 时使用。

    注意：内部仍持有同一把 per-task 锁，请保证 ``work`` 内不会再去抢同一 task 的锁。
    """

    lock = await stream._lock_for(task_id)  # noqa: SLF001 - 内部协作
    async with lock:
        current = await stream.get(task_id)
        if current is None:
            current = make_initial_snapshot(task_id, "", "")
        result = await work(current)
        snapshot = result if isinstance(result, dict) else current
        think, answer = recompute_aggregates(snapshot)
        snapshot["think"] = think
        snapshot["answer"] = answer
        snapshot["updatedAt"] = time.time()
        await stream._client.set(  # noqa: SLF001
            stream.key_for(task_id), snapshot, ttl_seconds=stream._ttl  # noqa: SLF001
        )
        return snapshot

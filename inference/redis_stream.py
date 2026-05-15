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
        # skillsDone：与 ``skills`` 列表分离的完成标记。``set_skills`` 一次性写入结果
        # 列表，但门控聚合需要一个独立 bool；``skillsEnabled=False`` 时 pipeline
        # 会在跳过 skills 阶段时直接把它设为 True，保证后续段能被解锁。
        "skillsDone": False,
        # react.usedHeadings：react_loop 在确定要进 final 轮之前回填的"真正进入
        # prompt 的知识章节叶子标题"列表（原样形如 ``1.1_章节名``，去重保序）。
        # 仅供 SSE think 聚合时渲染【引用知识章节】段；不影响给 LLM 的 prompt。
        #
        # react.intermediateLocked / react.finalLocked：SSE think 聚合的"段间锁"，
        # 用来保证前端拿到的 think 字段是 append-only：
        # - intermediateLocked：所有中间轮跑完、即将进 final/forced_final 时由
        #   react_loop 翻成 True。翻 True 之后 ``recompute_aggregates`` 才把
        #   ``chunks[-1]`` 当作 final.think 拼到聚合 think 里；翻 True 之前所有
        #   chunk 都按中间轮处理，避免"动态 final"导致前面已经吐过的内容被改写。
        # - finalLocked：final 轮整体收尾后翻 True。翻 True 之后聚合 think 末尾
        #   才追加【引用知识章节】段，避免段位置随 chunks 增长而漂移。
        # 中间轮自身可见性还要求 ``preview.done && skillsDone``，由聚合规则统一判定。
        "react": {
            "round": 0,
            "chunks": [],
            "usedHeadings": [],
            "intermediateLocked": False,
            "finalLocked": False,
        },
        # topicLocate：仅 /api/inference/stream 在用户未指定 policyId 时会写入，
        # 用于把外部【专题Know How定位】SSE 的 reasoning 流式转发到 think 字段，
        # 并以 ``###【专题Know How定位】\n{ywzt}`` 收口。各字段语义：
        # - reasoning: 外部 SSE ``data.reasoning``（每帧全量字符串，覆盖式更新）
        # - ywzt:      外部 SSE finish 帧的业务专题名（命中时填）
        # - done:      finish 且唯一命中时为 True，触发 ywzt 收口段渲染
        # - skipped:   用户已传 policyId（或 V4 路径）时为 True，整段不展示
        # - refusal:   候选业务专题非唯一（0/多）时的拒答文案，由聚合规则覆盖到接口 answer
        "topicLocate": {
            "reasoning": "",
            "ywzt": "",
            "done": False,
            "skipped": False,
            "refusal": "",
        },
        "intermediateThinkEnabled": flag,
        "think": "",
        "answer": "",
        # answerable：本次任务是否可以继续/已经给出有效答案。三态语义：
        # - None（默认 / 未确定）：开启了【专题Know How定位】路径但还未拿到终态结论
        #   （topicLocate 既未 skipped 也未 done，refusal 也为空），此时 think/answer
        #   强制为空字符串，SSE 接力侧据此**完全不流出 snapshot**，避免把中间 reasoning
        #   过程暴露给前端；
        # - True：业务专题已唯一命中（topicLocate.done=True）或本次任务直接跳过专题定位
        #   （topicLocate.skipped=True，已传 policyId / V4 路径），推理可正常推进；
        # - False：第一步【专题Know How定位】命中拒答条件（0/多候选）或解析失败，
        #   此时 think/answer 强制为空字符串，前端应据此短路结束 SSE 接力。
        # 由 recompute_aggregates 根据 topicLocate.{skipped,done,refusal} 自动派生，
        # 写入路径无需感知。
        "answerable": None,
        "error": None,
        # verbose 模式下 task 启动后由 pipeline 包装层回填，
        # 对应 <project>/verbose_logs/<YYYYMMDD_HHMMSS>_<taskId>.jsonl。
        # 未开启 verbose 时保持 None。
        "logName": None,
        "createdAt": now,
        "updatedAt": now,
    }


# ---------------------------------------------------------------- 聚合规则

def recompute_aggregates(
    snapshot: Snapshot,
    intermediate_think_enabled: Optional[bool] = None,
) -> tuple[str, str, Optional[bool]]:
    """根据 react/preview/topicLocate 子结构重算接口 ``think`` / ``answer`` / ``answerable``。

    **聚合的核心原则是"段间锁 + append-only"**：前端拿到的 think 字段每帧都是从快照
    重算的全量字符串，因此聚合规则必须保证后段在所有前段稳定之前不被拼出，否则前端
    会看到"已经吐过的内容被改写"。具体做法是对每个段引入显式可见性门控：

    拼接顺序（块间以空行 ``\\n\\n`` 分隔）：

    1. **topicLocate**（仅 ``skipped=False`` 时输出，按需）：
       - 流式覆盖的 ``reasoning``；
       - finish 后命中且 ``ywzt`` 非空：``###【专题Know How定位】\\n\\t{ywzt}``。
    2. **preview**：``preview.think`` / ``preview.answer``（顺序固定、内部 append-only）。
    3. **react 中间轮**：可见性需要 ``preview.done && skillsDone``，否则不拼。
       开关关时仅取最后一个中间轮的 ``answer``（兼容旧规则，实际为空）；
       开关开时全部中间轮的 ``think + answer``。
       谁是"中间轮"取决于 ``react.intermediateLocked``：
       - ``intermediateLocked=False``：所有 chunk 都按中间轮看（final 还没确认）；
       - ``intermediateLocked=True``：``chunks[-1]`` 是 final，前面是中间轮。
    4. **react 最终轮**：仅 ``intermediateLocked=True`` 时拼出 ``final.think``,
       并把 ``final.answer`` 作为接口 ``answer``。
    5. **【引用知识章节】**：仅 ``finalLocked=True`` 时追加到 think 末尾，避免段位置
       随 chunks 增长 / set_used_headings 早写而漂移。

    所有 lock 字段由 react_loop / pipeline / preview / skills_runner 在阶段切换时显式翻
    True；``preview_enabled=False`` / ``skills_enabled=False`` 时 pipeline 立即把对应
    done 翻 True，从而不会卡住后续段。

    ``answerable`` 三态：

    - ``False``：``topicLocate.refusal`` 非空（候选业务专题 0/多 命中，或外部 SSE 解析
      失败等定位异常场景），整条任务被判定为不可应答，``think`` / ``answer`` 强制返回
      空字符串，调用方据此立即结束 SSE 接力；
    - ``None``：未确定 —— 开启了【专题Know How定位】路径但还在跑（既未 ``skipped`` 也
      未 ``done``，且 ``refusal`` 为空），``think`` / ``answer`` 强制返回空字符串，
      SSE 接力侧应**完全跳过 snapshot 推送**，避免把中间 reasoning 暴露给前端；
    - ``True``：正常路径，``topicLocate.skipped=True`` 或 ``topicLocate.done=True``，
      此时按上述拼接规则正常输出 ``think`` / ``answer``。
    """

    if intermediate_think_enabled is None:
        intermediate_think_enabled = bool(
            snapshot.get("intermediateThinkEnabled", config.REACT_INTERMEDIATE_THINK_ENABLED)
        )

    preview = snapshot.get("preview") or {}
    react = snapshot.get("react") or {}
    chunks = react.get("chunks") or []
    topic_locate = snapshot.get("topicLocate") or {}

    # 段间锁的"字段缺失"兼容策略：缺失 → 默认 True（按"已锁定"看待）。
    # 这是为了向后兼容升级前已存在于 redis 的旧 task snapshot——它们没有这些字段,
    # 一旦按 False 看待会让 stream 接力查看时所有 react 段都消失。
    # ``make_initial_snapshot`` 给新 task 已显式写 False，所以新 task 走正常门控；
    # 唯一拿到默认 True 的就是"升级前已生成的存量 snapshot"。
    # 注意 ``preview.done`` 历史以来都在 schema 中（旧版本也写），保留 default=True
    # 仅作极端兜底。
    preview_done = bool(preview.get("done", True))
    skills_done = bool(snapshot.get("skillsDone", True))
    intermediate_locked = bool(react.get("intermediateLocked", True))
    final_locked = bool(react.get("finalLocked", True))

    parts: list[str] = []

    # 1) topicLocate 段：仅在未跳过时尝试输出。reasoning 先到、ywzt 收口后到。
    if not topic_locate.get("skipped"):
        tl_reason = (topic_locate.get("reasoning") or "").strip()
        if tl_reason:
            parts.append(tl_reason)
        if topic_locate.get("done"):
            ywzt = (topic_locate.get("ywzt") or "").strip()
            if ywzt:
                parts.append(f"【专题Know How定位】\n\t{ywzt}")

    # 2) preview 段：始终展示（preview 是 react 之前的最前段，内部 append-only，
    # 不会让前面的 topicLocate 段发生位置变化）。
    p_think = (preview.get("think") or "").strip()
    p_answer = (preview.get("answer") or "").strip()
    if p_think:
        parts.append(p_think)
    if p_answer:
        parts.append(p_answer)

    answer = ""
    if chunks:
        # 用 intermediate_locked 锁定 final 身份：未锁定时全部 chunk 都按中间轮看
        # （此时即使展示，也都是中间轮，前端不会看到"final 段先出现又被改回中间轮"
        # 的翻转）；锁定后 chunks[-1] 才被确认为 final。
        if intermediate_locked:
            final_chunk = chunks[-1] or {}
            mid_chunks = chunks[:-1]
        else:
            final_chunk = {}
            mid_chunks = chunks

        # 3) 中间轮：preview/skills 都完成才进入聚合。
        # 否则即便已经在写入 react.chunks 也不外露——等 preview/skills done 后,
        # 中间轮已累积的内容会作为整段一次性"插入到 preview 完整结果下方",
        # 之后随 react 继续流而 append-only 增长。
        if preview_done and skills_done and mid_chunks:
            if intermediate_think_enabled:
                for c in mid_chunks:
                    c = c or {}
                    c_think = (c.get("think") or "").strip()
                    c_answer = (c.get("answer") or "").strip()
                    if c_think:
                        parts.append(c_think)
                    if c_answer:
                        parts.append(c_answer)
            else:
                # 旧规则：仅最后一个中间轮的 answer。
                # 注意：现行实现里中间轮的 chunk.answer 始终为空（answer_as_verdict
                # 路径不写 answer），此分支实际不会产出可见文本；保留以兼容旧路径。
                last_mid = mid_chunks[-1] or {}
                lm_answer = (last_mid.get("answer") or "").strip()
                if lm_answer:
                    parts.append(lm_answer)

        # 4) final.think：必须 intermediateLocked 才解锁；未锁定时 chunks[-1] 算
        # 中间轮，已在上一段按中间轮规则处理过（或被门控不显示）。
        if intermediate_locked:
            final_think = (final_chunk.get("think") or "").strip()
            final_answer = (final_chunk.get("answer") or "").strip()
            if final_think:
                parts.append(final_think)
            answer = final_answer

            # 5) 引用章节：final 整体收尾后才追加到 think 末尾，避免 chunks 增长
            # / set_used_headings 早写引起的段位置漂移。
            if final_locked:
                used_headings = react.get("usedHeadings") or []
                if used_headings:
                    cited = "\n".join(str(h) for h in used_headings if h)
                    if cited:
                        parts.append(f"【引用知识章节】\n{cited}")

    # 拒答 / 解析错误：业务专题识别没拿到可用 policyId，本次推理不可应答。
    # 整条任务退化为"think 空 + answer 空 + answerable=False"，前端据此立即收尾,
    # 不再渲染前面累积的 topicLocate.reasoning 等任何中间段。
    refusal = (topic_locate.get("refusal") or "").strip()
    if refusal:
        return "", "", False

    # 未确定：开启了【专题Know How定位】路径但仍在 reasoning 阶段（skipped=False 且
    # done=False，refusal 也为空）。此时本次任务尚未确认是否能继续推理，think/answer
    # 强制置空，answerable=None，由调用方在 SSE 端跳过整帧推送，避免把中间过程外泄。
    if not topic_locate.get("skipped") and not topic_locate.get("done"):
        return "", "", None

    think = "\n\n".join(parts).strip()
    return think, answer, True


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
            think, answer, answerable = recompute_aggregates(snapshot)
            snapshot["think"] = think
            snapshot["answer"] = answer
            snapshot["answerable"] = answerable
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

    async def set_log_name(self, task_id: str, log_name: Optional[str]) -> Snapshot:
        """回写 verbose 日志文件名。pipeline 包装层开启 verbose session 后调用。"""

        def _mut(s: Snapshot) -> None:
            s["logName"] = log_name

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

    async def set_skills_done(self, task_id: str, done: bool = True) -> Snapshot:
        """翻 ``skillsDone`` 门控。skills 阶段（``run_skills``）整体收尾时调用；
        ``skills_enabled=False`` 时 pipeline 应在进 react 之前立即翻 True 以解锁
        后续段的可见性。
        """

        def _mut(s: Snapshot) -> None:
            s["skillsDone"] = bool(done)

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

    async def set_used_headings(
        self,
        task_id: str,
        headings: list[str],
    ) -> Snapshot:
        """react_loop 在确定要进 final 轮之前回填"真正进入 prompt 的章节叶子标题"。

        ``headings`` 已由调用方去重保序，元素形如 ``1.1_章节名``。聚合规则会在
        ``react.finalLocked=True`` 之后才把这些标题渲染成 ``###【引用知识章节】`` 段
        追加到 think 末尾（仅影响 SSE think 字段，不影响给 LLM 的 prompt）。
        """

        def _mut(s: Snapshot) -> None:
            react = s.setdefault("react", {"round": 0, "chunks": []})
            react["usedHeadings"] = [str(h) for h in (headings or []) if h]

        return await self.update(task_id, _mut)

    async def set_react_intermediate_locked(
        self,
        task_id: str,
        locked: bool = True,
    ) -> Snapshot:
        """翻 ``react.intermediateLocked`` 门控。react_loop 在所有中间轮跑完、即将进
        final/forced_final 之前调用：

        - 翻 True 之前：所有 chunk 都按"中间轮"看，``chunks[-1]`` 即便已被写入也不会
          被聚合当成 final.think，避免下一轮 chunk 新增时引起的"final 翻转"。
        - 翻 True 之后：``chunks[-1]`` 被确认为 final，``final.think`` 才出现在
          聚合 think 中、``final.answer`` 才进入接口 ``answer`` 字段。
        """

        def _mut(s: Snapshot) -> None:
            react = s.setdefault("react", {"round": 0, "chunks": []})
            react["intermediateLocked"] = bool(locked)

        return await self.update(task_id, _mut)

    async def set_react_final_locked(
        self,
        task_id: str,
        locked: bool = True,
    ) -> Snapshot:
        """翻 ``react.finalLocked`` 门控。react_loop 在 final 轮整体收尾后调用：

        翻 True 之后聚合 think 末尾才追加 ``###【引用知识章节】`` 段；翻 True 之前
        即便 ``usedHeadings`` 已经写入也不会渲染，避免段位置随 chunks 增长漂移。
        """

        def _mut(s: Snapshot) -> None:
            react = s.setdefault("react", {"round": 0, "chunks": []})
            react["finalLocked"] = bool(locked)

        return await self.update(task_id, _mut)

    # ----- topic locate ---------------------------------------------
    #
    # 仅 /api/inference/stream 在用户未指定 policyId 时调用。所有写法都走 update,
    # 让 recompute_aggregates 自动把 topicLocate 段拼到接口 think/answer 上。

    async def append_topic_locate_reasoning(
        self,
        task_id: str,
        reasoning_full: str,
    ) -> Snapshot:
        """**覆盖式**写入外部【专题Know How定位】SSE 的 ``data.reasoning``。

        外部接口每帧推送的就是当前累积的全量 reasoning 字符串（非增量 delta），
        这里直接覆盖最自然，且能避免重复拼接放大体积。
        """

        def _mut(s: Snapshot) -> None:
            tl = s.setdefault("topicLocate", {
                "reasoning": "", "ywzt": "", "done": False,
                "skipped": False, "refusal": "",
            })
            tl["reasoning"] = reasoning_full or ""

        return await self.update(task_id, _mut)

    async def set_topic_locate_done(
        self,
        task_id: str,
        ywzt: str,
    ) -> Snapshot:
        """外部 SSE finish 帧、且候选业务专题唯一命中时调用，触发 ywzt 收口段渲染。"""

        def _mut(s: Snapshot) -> None:
            tl = s.setdefault("topicLocate", {
                "reasoning": "", "ywzt": "", "done": False,
                "skipped": False, "refusal": "",
            })
            tl["ywzt"] = (ywzt or "").strip()
            tl["done"] = True

        return await self.update(task_id, _mut)

    async def set_topic_locate_skipped(self, task_id: str) -> Snapshot:
        """已传 policyId（或 V4 路径）时调用：整段【专题Know How定位】不展示。"""

        def _mut(s: Snapshot) -> None:
            tl = s.setdefault("topicLocate", {
                "reasoning": "", "ywzt": "", "done": False,
                "skipped": False, "refusal": "",
            })
            tl["skipped"] = True

        return await self.update(task_id, _mut)

    async def set_topic_locate_refusal(
        self,
        task_id: str,
        refusal: str,
    ) -> Snapshot:
        """候选业务专题非唯一/定位异常时写入拒答文案，由聚合规则覆盖到接口 answer。"""

        def _mut(s: Snapshot) -> None:
            tl = s.setdefault("topicLocate", {
                "reasoning": "", "ywzt": "", "done": False,
                "skipped": False, "refusal": "",
            })
            tl["refusal"] = (refusal or "").strip()

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
        think, answer, answerable = recompute_aggregates(snapshot)
        snapshot["think"] = think
        snapshot["answer"] = answer
        snapshot["answerable"] = answerable
        snapshot["updatedAt"] = time.time()
        await stream._client.set(  # noqa: SLF001
            stream.key_for(task_id), snapshot, ttl_seconds=stream._ttl  # noqa: SLF001
        )
        return snapshot

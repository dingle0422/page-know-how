"""Step4：分块 ReAct 多轮推理主循环。

关键设计：

- 输入是已检索好的 ``KnowledgeChunk`` 列表（由 pipeline 从 hybrid_search 取得后传入）。
  本模块**不直接调用检索**，让单元测试与离线回放更容易。
- 按 ``CHUNK_SIZE`` 把 chunk 重新打包成"轮组"，每一轮喂给模型一组证据。
- 每一轮：
  1. 从 redis 重读 preview / skills 快照（捕获最新进展）；
  2. 通过 :func:`inference.prompts.select_react_prompt` 选择 prompt：
     - 中间轮：``REACT_INTERMEDIATE_*``，仅产 ``<think>+<verdict>``，禁止 ``<answer>``；
     - 最终轮：纯 ``CORPUS_SYSTEM_PROMPT`` + ``CORPUS_USER_PROMPT``，产 ``<think>+<answer>``。
  3. 通过 :func:`inference.llm_stream.chat_stream` 流式拉取，router 拆 ``<think>``
     / ``<answer>`` / ``<verdict>``，对应增量写回 ``react.chunks[round_idx]``；
  4. 读取 ``router.verdict``：``complete`` 或已是最终轮则结束。
- ``REACT_INTERMEDIATE_THINK_ENABLED`` 默认关：
  - 中间轮 ``enable_thinking=False`` 且 ``chunk.think`` 不写（聚合规则也不会取到）；
  - 仅最终轮 ``enable_thinking=True``，think + answer 一起进 chunk。
  - 开关开时：所有轮都 ``enable_thinking=True``，每轮 think+answer 都进 chunk，
    由 :func:`inference.redis_stream.recompute_aggregates` 统一计算 SSE 聚合 think。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from typing import Awaitable, Callable, Optional

from knowledge_core.chunk_builder import KnowledgeChunk
from utils.verbose_logger import (
    is_session_active,
    log_llm_call,
    log_llm_error,
    step_scope,
)

from . import config
from .llm_stream import StreamTagRouter, chat_stream
from .prompts import select_react_prompt
from .redis_stream import RedisStream

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- chunk 打包

_CHUNK_SEPARATOR = "\n\n---\n\n"


# ---------------------------------------------------------------- reSearch 协议解析
#
# 仅 ``re_search_enabled=True`` 时使用。中间轮 <answer> 内嵌
# <completion>/<action>/<search-query> 三个子标签，本节负责从模型流式产出的
# *原始 answer 文本*（含子标签字面量）里把决策结构解析出来。
# 解析口径与回退策略见 react_research_mode_upgrade 计划：
#   1. 优先解析新子标签；
#   2. completion=complete 时直接忽略 action/search-query；
#   3. 子标签缺失 → 退回旧协议（整段 answer 当 complete|incomplete）；
#   4. 连 completion 都无法可靠解析 → 强制按 paginate 执行（不中断主流程）。

_COMPLETION_RE = re.compile(r"<completion>(.*?)</completion>", re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
_SEARCH_QUERY_RE = re.compile(
    r"<search-query>(.*?)</search-query>", re.IGNORECASE | re.DOTALL
)


def _normalize_completion(text: str) -> Optional[str]:
    """把任意文本归一化为 ``complete`` / ``incomplete`` / None（无法判定）。

    宽松匹配，兼容大小写漂移与尾随标点：先判 incomplete（它含 complete 子串，
    必须优先），再判 complete。
    """

    t = (text or "").strip().lower().strip(" \t\n。.,;；:!?\"'`")
    if not t:
        return None
    if "incomplete" in t:
        return "incomplete"
    if "complete" in t:
        return "complete"
    return None


def _normalize_action(text: str) -> str:
    """把 <action> 文本归一化为 ``research`` / ``paginate``（默认 paginate）。"""

    t = (text or "").strip().lower()
    if "research" in t:
        return "research"
    return "paginate"


def parse_research_answer(full_answer: str) -> dict:
    """解析 reSearch 中间轮的原始 answer，返回决策 dict。

    返回结构：``{"completion": "complete"|"incomplete", "action": "paginate"|"research",
    "search_query": str}``。

    解析规则（与计划一致）：

    - 先尝试 ``<completion>`` 子标签；缺失时退回旧协议——把整段 answer（去掉可能存在的
      其它子标签）按 ``complete|incomplete`` 解释；
    - 二者都解析不出可靠 completion 时，强制按 ``incomplete + paginate`` 兜底（不抛错）；
    - completion=complete 时强制忽略 action/search-query（即便模型违规输出）；
    - completion=incomplete 时解析 action；action=research 且 search-query 非空白才真正
      触发 research，否则降级 paginate。
    """

    raw = full_answer or ""

    completion: Optional[str] = None
    m = _COMPLETION_RE.search(raw)
    if m:
        completion = _normalize_completion(m.group(1))
    if completion is None:
        # 退回旧协议：把整段 answer 当 verdict 解释（去掉子标签噪声后再判）。
        stripped = _ACTION_RE.sub(" ", raw)
        stripped = _SEARCH_QUERY_RE.sub(" ", stripped)
        completion = _normalize_completion(stripped)

    if completion is None:
        # 连最小信号都拿不到：兜底按"还没结束 + 翻页"处理。
        return {"completion": "incomplete", "action": "paginate", "search_query": ""}

    if completion == "complete":
        # complete 时忽略 action/search-query。
        return {"completion": "complete", "action": "paginate", "search_query": ""}

    # incomplete：解析 action / search-query。
    action = "paginate"
    am = _ACTION_RE.search(raw)
    if am:
        action = _normalize_action(am.group(1))

    search_query = ""
    if action == "research":
        sm = _SEARCH_QUERY_RE.search(raw)
        if sm:
            # 单行化：折叠所有空白为单个空格。
            search_query = " ".join((sm.group(1) or "").split())
        if not search_query:
            # research 但没给出有效检索串 → 降级 paginate。
            action = "paginate"

    return {"completion": "incomplete", "action": action, "search_query": search_query}


def _chunk_dedup_key(chunk: KnowledgeChunk) -> str:
    """生成 chunk 的去重键。

    KnowledgeChunk 没有跨多次检索稳定的全局 id（``index`` 仅在单次返回列表内有效），
    因此用规范化正文（trim + 折叠空白）的 md5 作为稳定标识——同一知识块在不同检索
    结果里的正文一致，hash 即一致。
    """

    content = (getattr(chunk, "content", "") or "")
    norm = " ".join(content.split())
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


class _ReactChannelBuffer:
    """单轮 react 流式的 think / answer 累积缓冲。

    与 :class:`inference.preview._ChannelBuffer` 等价，专门给
    :func:`_stream_react_round` 内的批量 flush writer 使用：

    - LLM SSE 每个 delta 不再直接触发一次 ``RedisStream.update``
      （那条路径每次都要等一次远程 redis_server 的 ``GET+SET`` RTT）；
    - 改为按 :data:`config.REACT_TPS` 周期把缓冲里累积的整段 think/answer
      一次性 ``append_react_chunk_delta``。

    详见 :data:`config.REACT_TPS` 的 docstring（解释了为什么改、阈值怎么取）。
    """

    __slots__ = ("think", "answer")

    def __init__(self) -> None:
        self.think: list[str] = []
        self.answer: list[str] = []

    def push_think(self, delta: str) -> None:
        if delta:
            self.think.append(delta)

    def push_answer(self, delta: str) -> None:
        if delta:
            self.answer.append(delta)

    def take(self) -> tuple[str, str]:
        t = "".join(self.think)
        a = "".join(self.answer)
        self.think.clear()
        self.answer.clear()
        return t, a

    def empty(self) -> bool:
        return not self.think and not self.answer


def pack_chunks_by_size(
    chunks: list[KnowledgeChunk],
    *,
    chunk_size: int = config.CHUNK_SIZE,
) -> list[str]:
    """把 KnowledgeChunk 列表按字符上限打包成"轮组"文本。

    - 单 chunk 超 ``chunk_size`` 时单独成组，不强切，保留语义完整性。
    - 多个小 chunk 累计到接近上限时切组。
    - 返回 ``list[str]``，每项就是一轮要喂给模型的证据全文。
    """

    if not chunks:
        return []
    sep_len = len(_CHUNK_SEPARATOR)
    groups: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for c in chunks:
        text = (c.content or "").strip()
        if not text:
            continue
        addition = len(text) + (sep_len if current else 0)
        if current and current_len + addition > chunk_size:
            groups.append(current)
            current = []
            current_len = 0
            addition = len(text)
        current.append(text)
        current_len += addition
    if current:
        groups.append(current)
    return [_CHUNK_SEPARATOR.join(g) for g in groups]


def _pack_chunks_with_indices(
    chunks: list[KnowledgeChunk],
    *,
    chunk_size: int = config.CHUNK_SIZE,
) -> list[list[int]]:
    """与 :func:`pack_chunks_by_size` 同算法，但返回每组对应的 chunk 索引列表。

    用于 :func:`run` 在确定 final 轮之前反推"真正进入 prompt 的 chunks 集合",
    把这些 chunk 的 ``heading_paths`` 叶子写到 redis 快照的 ``react.usedHeadings``,
    供 SSE think 聚合渲染【引用知识章节】段。

    与 :func:`pack_chunks_by_size` 的差异**仅**在返回形态：

    - 这里不做字符串拼接（避免重复内存）；
    - 跳过的 ``empty`` chunk（content 为空）同样跳过，确保索引语义与上面那套一致。
    """

    if not chunks:
        return []
    sep_len = len(_CHUNK_SEPARATOR)
    groups: list[list[int]] = []
    current: list[int] = []
    current_len = 0
    for i, c in enumerate(chunks):
        text = (getattr(c, "content", "") or "").strip()
        if not text:
            continue
        addition = len(text) + (sep_len if current else 0)
        if current and current_len + addition > chunk_size:
            groups.append(current)
            current = []
            current_len = 0
            addition = len(text)
        current.append(i)
        current_len += addition
    if current:
        groups.append(current)
    return groups


def _collect_used_headings(
    chunks: list[KnowledgeChunk],
    used_indices: list[int],
) -> list[str]:
    """从命中 chunks 的 ``heading_paths`` 中收集叶子标题，去重保序。

    返回元素形如 ``1.1_章节名``（按用户口径直接用叶子原文）。同名叶子（即使来自
    不同 chunk）只保留首次出现，保证渲染时的稳定顺序。
    """

    seen: set[str] = set()
    out: list[str] = []
    for i in used_indices:
        if i < 0 or i >= len(chunks):
            continue
        for path in (getattr(chunks[i], "heading_paths", None) or []):
            if not path:
                continue
            leaf = path[-1]
            if isinstance(leaf, str) and leaf and leaf not in seen:
                seen.add(leaf)
                out.append(leaf)
    return out


# ---------------------------------------------------------------- 单轮流式


async def _stream_react_round(
    task_id: str,
    round_idx: int,
    redis_stream: RedisStream,
    *,
    system_prompt: str,
    user_prompt: str,
    vendor: str,
    model: str,
    is_last_chunk: bool,
    enable_thinking: bool,
    store_think_to_chunk: bool,
    answer_as_verdict: bool,
    research_mode: bool = False,
    flush_tps: int = config.REACT_TPS,
) -> tuple[str, str, str]:
    """跑一轮，返回 ``(verdict, full_think, full_answer)``。

    - ``research_mode=True``（仅 reSearch 中间轮）：``<answer>`` 内嵌
      ``<completion>/<action>/<search-query>`` 子标签。本函数**不**解析这些子标签，
      只把原始 answer 文本（含子标签字面量）通过 ``full_answer`` 原样返回，交由
      :func:`parse_research_answer` 在轮末解析；``verdict`` 返回空串（不使用）。
      与普通中间轮一致：``answer_as_verdict=True`` 时 answer 不写入 chunk，
      因此子标签不会流式展示给用户。

    - ``store_think_to_chunk=False`` 时，本轮模型产出的 think 仅累积在内存中（用于
      下一轮的 ``prev_think``），不写入 ``react.chunks[round_idx].think``，对应默认
      模式下中间轮 ``chunk.think=""`` 的约定。
    - ``answer_as_verdict=False``（最终轮）：``<answer>`` 是真正的用户答案，按增量
      写到 ``chunk.answer``，``verdict`` 取自 ``router.verdict``（旧 <verdict> 标签）。
    - ``answer_as_verdict=True``（中间轮，新 prompt 协议）：``<answer>`` 内容固定是
      ``complete`` / ``incomplete`` 的裁决字符串，仅作 verdict 使用，**不** 写到
      ``chunk.answer``（否则会被 SSE 聚合误当成真实答案显示）；``verdict`` 直接取自
      ``full_answer.strip().lower()``。

    写 Redis 走"批量 flush"模式：LLM 每个 SSE delta 仅 ``push`` 到内存 buffer，
    后台 ``writer_loop`` 按 ``1/flush_tps`` 秒周期把累积的 think/answer 各 ``append``
    一次到 ``react.chunks[round_idx]``。这样把单 token 速度从"必须等一次
    远程 redis_server 的 GET+SET RTT"放宽到"每窗口一次 GET+SET"，避免远程
    HTTP RTT 反压到 LLM SSE 接收链路（详见 :data:`config.REACT_TPS` docstring）。
    ``store_think_to_chunk`` / ``answer_as_verdict`` 的语义完全保持原样：
    仅控制对应通道是否 ``push`` 进 buffer，flush 路径不变。
    """

    interval = 1.0 / max(int(flush_tps), 1)
    buffer = _ReactChannelBuffer()
    stop = asyncio.Event()

    async def _flush_once() -> None:
        """把 buffer 里累积的 think/answer 各 append 一次到 redis 快照。

        think / answer 分两次 ``append_react_chunk_delta`` 调用：每次走一遍
        ``RedisStream.update`` 的 ``GET → mutate → SET`` 路径。两次合并到一次
        会减半 RTT，但需要新增 redis 接口；暂时保留两次调用以避免侵入
        ``RedisStream`` 公共面。flush 频率本身已经被 ``flush_tps`` 钉死，
        总 RPS 影响有限。
        """

        think_delta, answer_delta = buffer.take()
        if think_delta:
            try:
                await redis_stream.append_react_chunk_delta(
                    task_id, round_idx, "think", think_delta,
                    is_last_chunk=is_last_chunk,
                )
            except Exception as e:
                # 一笔失败不影响后续：下游会从 redis 丢失对应增量，但不会卡死流。
                logger.warning(
                    "[InferenceReact] task=%s round=%d flush think 失败: %s",
                    task_id, round_idx, e,
                )
        if answer_delta:
            try:
                await redis_stream.append_react_chunk_delta(
                    task_id, round_idx, "answer", answer_delta,
                    is_last_chunk=is_last_chunk,
                )
            except Exception as e:
                logger.warning(
                    "[InferenceReact] task=%s round=%d flush answer 失败: %s",
                    task_id, round_idx, e,
                )

    async def writer_loop() -> None:
        """周期性把 buffer 里累积的 think/answer flush 到 redis。

        - 用 ``asyncio.Event`` + ``asyncio.wait_for(stop.wait(), timeout=interval)``
          实现"要么等到 ``interval`` 自然超时（窗口到了）、要么被 ``stop`` 立刻唤醒
          （chat_stream 结束）"。``CancelledError`` 走 finally 兜底再 flush 一次。
        - 兜底 flush 必跑：即便 chat_stream 异常退出，最后窗口里累积但未 flush 的
          delta 仍会落盘，避免出现"模型已经吐了但 redis 看不到"的截断。
        """

        try:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
                await _flush_once()
        except asyncio.CancelledError:
            pass
        finally:
            # 退出前再 flush 一次剩余增量，保证最后窗口的 delta 不丢。
            await _flush_once()

    writer_task = asyncio.create_task(
        writer_loop(), name=f"react-writer:{task_id}:{round_idx}"
    )

    full_think_parts: list[str] = []
    full_answer_parts: list[str] = []
    router = StreamTagRouter()

    def _emit_think(s: str) -> None:
        if not s:
            return
        full_think_parts.append(s)
        if store_think_to_chunk:
            buffer.push_think(s)

    def _emit_answer(s: str) -> None:
        if not s:
            return
        full_answer_parts.append(s)
        # 中间轮 <answer> 内容只是 complete/incomplete 的 verdict 字符串，
        # 不能写到 chunk.answer——否则 SSE 聚合会把它当成最终答案吐给前端。
        if not answer_as_verdict:
            buffer.push_answer(s)

    verbose_on = is_session_active()
    t0 = time.time() if verbose_on else None
    stream_error: Exception | None = None

    try:
        async for channel, delta in chat_stream(
            user_prompt,
            vendor=vendor,
            model=model,
            system=system_prompt,
            enable_thinking=enable_thinking,
        ):
            if channel == "think":
                # 来自 reasoning_content：直接走 think
                _emit_think(delta)
            else:
                # 来自 content：可能夹 <think>/<answer>/<verdict> 标签，过 router
                router.feed(delta, on_think=_emit_think, on_answer=_emit_answer)
    except Exception as e:
        stream_error = e
        raise
    finally:
        stop.set()
        try:
            await writer_task
        except Exception as e:
            logger.warning(
                "[InferenceReact] task=%s round=%d writer 退出异常: %s",
                task_id, round_idx, e,
            )

        if verbose_on:
            think_so_far = "".join(full_think_parts).strip()
            answer_so_far = "".join(full_answer_parts).strip()
            verdict_so_far = (router.verdict or "").strip()
            response_text = (
                f"<think>\n{think_so_far}\n</think>\n<answer>\n{answer_so_far}\n</answer>"
                + (f"\n<verdict>{verdict_so_far}</verdict>" if verdict_so_far else "")
            )
            elapsed_ms = int((time.time() - t0) * 1000) if t0 is not None else None
            try:
                if stream_error is None:
                    log_llm_call(
                        prompt=user_prompt,
                        response=response_text,
                        system=system_prompt,
                        vendor=vendor,
                        model=model,
                        elapsed_ms=elapsed_ms,
                        extra={
                            "stage": "inference_react_round",
                            "task_id": task_id,
                            "round_idx": round_idx,
                            "is_last_chunk": is_last_chunk,
                            "enable_thinking": enable_thinking,
                            "store_think_to_chunk": store_think_to_chunk,
                            "verdict": verdict_so_far,
                        },
                    )
                else:
                    log_llm_error(
                        prompt=user_prompt,
                        error=f"{type(stream_error).__name__}: {stream_error}",
                        system=system_prompt,
                        vendor=vendor,
                        model=model,
                        elapsed_ms=elapsed_ms,
                        extra={
                            "stage": "inference_react_round",
                            "task_id": task_id,
                            "round_idx": round_idx,
                            "is_last_chunk": is_last_chunk,
                            "partial_think": think_so_far[:500],
                            "partial_answer": answer_so_far[:500],
                        },
                    )
            except Exception as log_e:
                logger.debug(
                    "[InferenceReact] verbose 落盘失败（忽略）: %s", log_e
                )

    full_think = "".join(full_think_parts).strip()
    full_answer = "".join(full_answer_parts).strip()
    if research_mode:
        # reSearch 中间轮：保留原始 answer（含 <completion>/<action>/<search-query>
        # 子标签字面量），由 react_loop.run 在轮末用 parse_research_answer 解析；
        # verdict 在此模式下无意义，返回空串占位。answer 不写 chunk 的语义已由
        # answer_as_verdict 路径保证。
        verdict = ""
    elif answer_as_verdict:
        # 中间轮：<answer> 内容就是裁决。先按 lower 归一化，再宽松匹配 complete,
        # 避免模型偶尔输出 "Complete." / "complete." 等带尾标点/大小写漂移的情况。
        candidate = full_answer.strip().lower().strip(" \t\n。.,;；:!?\"'`")
        if "complete" in candidate and "incomplete" not in candidate:
            verdict = "complete"
        elif "incomplete" in candidate:
            verdict = "incomplete"
        else:
            verdict = candidate
        # 既然 answer 在中间轮承担 verdict 语义，外层就不必再当成真实答案缓存。
        full_answer = ""
    else:
        # 最终轮：兼容历史 <verdict> 标签（虽然新 prompt 已弃用，但 router 仍会捕获,
        # 偶发被老/外部模型使用时不致丢信息）；空值由 react_loop.run 兜底为 "complete"。
        verdict = (router.verdict or "").strip().lower()
    return verdict, full_think, full_answer


# ---------------------------------------------------------------- 主入口


async def run(
    task_id: str,
    question: str,
    chunks: list[KnowledgeChunk],
    redis_stream: RedisStream,
    *,
    vendor: str = "servyou",
    model: str = "deepseek-v3.2-1163259bcc6c",
    chunk_size: int = config.CHUNK_SIZE,
    max_rounds: int = config.REACT_MAX_ROUNDS,
    intermediate_think_enabled: Optional[bool] = None,
    bg_tasks: Optional[list[asyncio.Task]] = None,
    re_search_enabled: bool = False,
    policy_id: Optional[str] = None,
    top_n: int = config.TOP_N,
    top_m: int = config.TOP_M,
    search_fn: Optional[Callable[[str], Awaitable[list[KnowledgeChunk]]]] = None,
) -> dict:
    """运行 ReAct 主循环。返回 ``{"rounds": int, "verdict": str, "answer": str}``。

    ``re_search_enabled`` 控制两条互不侵入的路径：

    - ``False``（默认）：完全沿用下方的旧主循环（固定遍历 groups + 中间轮只判
      ``complete|incomplete`` + 翻页 / forced final），旧 prompt 与旧逻辑零改动。
    - ``True``：走 :func:`_run_research_loop` 的动作驱动状态机——中间轮用新 prompt
      产 ``<completion>/<action>/<search-query>``，``incomplete`` 时按 ``paginate`` 翻页
      或 ``research`` 触发一次新的 :func:`hybrid_search` 召回（新证据相对"已进过 ReAct
      的旧证据"做单向去重后替换证据集）。需要配套传入 ``policy_id``（检索用，通常是
      带 ``__cs`` 后缀的服务端表名）、``top_n`` / ``top_m``；``search_fn`` 可注入用于
      单测，缺省时懒加载 :func:`inference.retrieval.hybrid_search`。

    以下 docstring 描述的是 ``re_search_enabled=False`` 的旧主循环行为（保持不变）：

    流程：
    - 按 ``groups`` 顺序逐轮跑中间轮（``answer_as_verdict=True``）：
      ``<think>`` 累进 ``prev_think``，``<answer>`` 内容是 ``complete``/``incomplete`` 裁决。
    - 任意一轮裁决 ``complete``：立刻追加**一轮强制 final**（``is_last_chunk=True``），
      用同一组证据 + 累积 ``prev_think`` 让模型真正写出用户可见答案，然后退出。
    - 跑到最后一组证据仍未 ``complete``：最后一组本身就按 final 跑，正常收尾。
    - 入参 ``chunks`` 为空时直接以一轮 "无检索证据" 跑最终轮。

    SSE 聚合"段间锁"配合（详见 ``redis_stream.recompute_aggregates``）：

    - **进 final 轮（含 forced_final）之前**：先 ``await bg_tasks``（preview / skills 还在
      跑就等它们收尾，保证 final prompt 能读到完整 preview/skills 快照），再把
      ``react.intermediateLocked`` 翻 True 以解锁 final.think 段的可见性。
    - **final 轮 ``set_react_chunk_complete`` 之后**：把 ``react.finalLocked`` 翻 True,
      以解锁 think 末尾的 ``【引用知识章节】`` 段。

    ``bg_tasks`` 由 pipeline 注入（preview + skills 的 ``asyncio.Task``）。传 ``None``
    或空 list 时跳过 await，仅翻锁；调用方在 ``preview_enabled=False`` /
    ``skills_enabled=False`` 时应预先把对应 done 标志翻 True。
    """

    flag = (
        bool(intermediate_think_enabled)
        if intermediate_think_enabled is not None
        else bool(config.REACT_INTERMEDIATE_THINK_ENABLED)
    )

    groups = pack_chunks_by_size(chunks, chunk_size=chunk_size)
    # groups_indices：与 groups 一一对应的"原始 chunk 索引"列表，用于后续反推
    # used chunks → heading_paths，供 SSE think 聚合渲染【引用知识章节】段。
    # chunks 为空时 groups 会被填占位证据，groups_indices 保持空 list,
    # 此时 _collect_used_headings 自然返回空，aggregates 不渲染该段。
    groups_indices = _pack_chunks_with_indices(chunks, chunk_size=chunk_size)
    if not groups:
        groups = ["（本次未检索到任何证据，请基于通识做最终回答。）"]
    if max_rounds > 0 and len(groups) > max_rounds:
        # 超额时把尾部多余 group 合并为最后一组，确保至少跑一次"最终轮"。
        head = groups[: max_rounds - 1]
        tail = _CHUNK_SEPARATOR.join(groups[max_rounds - 1 :])
        groups = head + [tail]
        # 索引侧同步合并：head + 把尾部各 group 的 indices 拍平为一组。
        if groups_indices and len(groups_indices) > max_rounds:
            head_idx = groups_indices[: max_rounds - 1]
            tail_idx = [
                idx for grp in groups_indices[max_rounds - 1 :] for idx in grp
            ]
            groups_indices = head_idx + [tail_idx]

    await redis_stream.set_status(task_id, "reasoning")

    async def _prepare_final(final_round_idx: int) -> None:
        """进 final/forced_final 之前的统一收口（顺序至关重要）：

        1. **先 ensure final chunk**：把 ``chunks[final_round_idx]`` 占位空 chunk
           写到 redis。**必须先于翻 intermediateLocked**——否则翻锁瞬间
           ``chunks[-1]`` 还是上一中间轮 ``c(N-1)``，会被聚合规则误当成 final 输出,
           随后 _run_one_round 再次调用 ensure_react_chunk 又把它推回中间轮，
           前端会看到一次"final 段瞬时出现又消失"的翻转。``ensure_react_chunk``
           本身是幂等的，重复调用无副作用。
        2. **再 await bg_tasks**：等 preview/skills 收尾，保证 final prompt 拿到
           完整的 preview/skills 快照（这是用户对 final 提示词完整性的硬要求）。
        3. **最后翻 intermediateLocked=True**：解锁 final.think 段的 SSE 可见性。
           此时 ``chunks[-1]`` = 空 final chunk，聚合输出 final.think="" 不影响
           前面已展示的 mid_chunks，append-only 不破坏。

        任一步异常都不抛出——只记录日志：preview/skills 在自己的 finally 里已经
        兜底翻了对应 done 标志，bg_task 抛异常时也被 ``pipeline._await_or_log``
        吞掉。即便 lock 翻失败，也不能阻塞 final 推理；最差只是 SSE 端少展示
        final 段。
        """

        try:
            await redis_stream.ensure_react_chunk(task_id, final_round_idx)
        except Exception as e:
            logger.warning(
                "[InferenceReact] task=%s ensure_react_chunk(final=%d) 失败: %s",
                task_id, final_round_idx, e,
            )
        if bg_tasks:
            try:
                await asyncio.gather(*bg_tasks, return_exceptions=True)
            except Exception as e:
                logger.warning(
                    "[InferenceReact] task=%s 等待 preview/skills 收尾异常: %s",
                    task_id, e,
                )
        try:
            await redis_stream.set_react_intermediate_locked(task_id, True)
        except Exception as e:
            logger.warning(
                "[InferenceReact] task=%s set_react_intermediate_locked 失败: %s",
                task_id, e,
            )

    async def _lock_final() -> None:
        """final/forced_final 整体收尾后翻 ``react.finalLocked=True``,
        解锁 think 末尾的【引用知识章节】段。"""

        try:
            await redis_stream.set_react_final_locked(task_id, True)
        except Exception as e:
            logger.warning(
                "[InferenceReact] task=%s set_react_final_locked 失败: %s",
                task_id, e,
            )

    # 仅保留"上一轮"的思考摘要，避免跨轮无限膨胀。
    prev_think = ""
    last_verdict = ""
    last_answer = ""
    total_groups = len(groups)
    rounds_done = 0

    async def _run_one_round(
        *, round_idx: int, group_text: str, is_last_chunk: bool, prev_think_val: str,
        step_kind: str, research_mode: bool = False,
    ) -> tuple[str, str, str]:
        """跑单轮：占位 chunk、读快照、组装 prompt、流式调模型、收尾标记。

        ``step_kind`` 用于 verbose 日志的 step label 后缀（``intermediate`` / ``final`` /
        ``final_forced`` / ``research``），便于在 jsonl 中区分各类轮次。

        ``research_mode=True``（仅 reSearch 中间轮）：选 research 版中间轮 prompt，
        并让 ``_stream_react_round`` 原样返回 answer（含 <completion>/<action>/
        <search-query> 子标签），由调用方用 :func:`parse_research_answer` 解析。
        """

        await redis_stream.ensure_react_chunk(task_id, round_idx)

        snapshot = await redis_stream.get(task_id) or {}
        preview_snapshot = snapshot.get("preview")
        skills_snapshot = snapshot.get("skills")

        system_prompt, user_prompt = select_react_prompt(
            is_last_chunk=is_last_chunk,
            question=question,
            evidence=group_text,
            prev_think=prev_think_val,
            preview=preview_snapshot,
            skills=skills_snapshot,
            re_search_enabled=research_mode,
        )

        enable_thinking_local = is_last_chunk or flag
        store_think_local = is_last_chunk or flag

        step_label = f"inference_react_{step_kind}_r{round_idx}"
        try:
            with step_scope(step_label, prompt_vars={
                "round_idx": round_idx,
                "is_last_chunk": is_last_chunk,
                "total_groups": total_groups,
                "step_kind": step_kind,
                "research_mode": research_mode,
            }):
                local_verdict, local_think, local_answer = await _stream_react_round(
                    task_id,
                    round_idx,
                    redis_stream,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    vendor=vendor,
                    model=model,
                    is_last_chunk=is_last_chunk,
                    enable_thinking=enable_thinking_local,
                    store_think_to_chunk=store_think_local,
                    # 中间轮的 <answer> 是 verdict / 决策字符串；最终轮的 <answer> 才是用户答案。
                    answer_as_verdict=not is_last_chunk,
                    research_mode=research_mode,
                )
        except Exception as exc:
            logger.exception(
                "[InferenceReact] task=%s round=%d (%s) 模型流式失败: %s",
                task_id, round_idx, step_kind, exc,
            )
            await redis_stream.set_react_chunk_complete(
                task_id, round_idx, complete=False, verdict="error"
            )
            raise

        await redis_stream.set_react_chunk_complete(
            task_id, round_idx, complete=True,
            verdict=local_verdict or ("complete" if is_last_chunk else "incomplete"),
        )
        return local_verdict, local_think, local_answer

    # ============================================================
    # reSearch=true：动作驱动状态机（paginate / research）
    # ============================================================
    if re_search_enabled:
        return await _run_research_loop(
            task_id=task_id,
            question=question,
            chunks=chunks,
            redis_stream=redis_stream,
            chunk_size=chunk_size,
            max_rounds=max_rounds,
            policy_id=policy_id,
            top_n=top_n,
            top_m=top_m,
            search_fn=search_fn,
            prepare_final=_prepare_final,
            lock_final=_lock_final,
            run_one_round=_run_one_round,
        )

    # ---- 主循环：依次跑各组证据 -----------------------------------
    forced_final_after_idx: int | None = None
    for round_idx, group_text in enumerate(groups):
        rounds_done = round_idx + 1
        is_last_chunk = round_idx == total_groups - 1
        # 正常收尾分支：本轮就是 final（最后一组），在 _run_one_round 调用前
        # 先把"全部进入过 prompt 的 chunks"对应的 heading 叶子写到 redis 快照,
        # 让 SSE think 在 finalLocked 翻 True 后能立即带上【引用知识章节】段。
        # 注意这里 used = 全部 indices：模型跑到最后一组自然收尾时所有 group 都被消费。
        if is_last_chunk:
            # 进 final 前的"段间锁"收口：先 ensure 占位 final chunk，再 await
            # preview/skills 收尾，最后翻 intermediateLocked。详见 _prepare_final
            # docstring 里关于顺序的解释（顺序不对会产生 final 段瞬时翻转）。
            await _prepare_final(round_idx)
            try:
                used_indices = [idx for grp in groups_indices for idx in grp]
                used_headings = _collect_used_headings(chunks, used_indices)
                await redis_stream.set_used_headings(task_id, used_headings)
            except Exception as e:
                # 写失败不阻塞推理：缺这段段头只影响 SSE 渲染，不影响最终 answer。
                logger.warning(
                    "[InferenceReact] task=%s 写 used_headings 失败（忽略）: %s",
                    task_id, e,
                )
        verdict, full_think, full_answer = await _run_one_round(
            round_idx=round_idx,
            group_text=group_text,
            is_last_chunk=is_last_chunk,
            prev_think_val=prev_think,
            step_kind="final" if is_last_chunk else "intermediate",
        )

        last_verdict = verdict
        last_answer = full_answer
        if full_think:
            # 下一轮只继承上一轮 think（由模型在本轮中自行完成降噪提炼），
            # 不再累积所有历史轮次，避免 prompt 体积持续膨胀。
            prev_think = full_think.strip()

        if is_last_chunk:
            # 最后一组本来就走 final，直接结束；last_answer 已是用户答案。
            # final 收尾，翻 finalLocked 解锁【引用知识章节】段。
            await _lock_final()
            break
        if verdict == "complete":
            # 模型在中间轮判定信息已足够；不再读新证据，下面追加一轮强制 final
            # 来真正写出 <answer>（中间轮的 <answer> 是裁决字符串，不是用户答案）。
            forced_final_after_idx = round_idx
            break

    # ---- 强制 final 兜底：中间轮判 complete 后追加一轮收尾 ----------
    if forced_final_after_idx is not None:
        forced_round_idx = forced_final_after_idx + 1
        rounds_done = forced_round_idx + 1
        logger.info(
            "[InferenceReact] task=%s round=%d 中间轮判 complete，"
            "追加强制 final 轮 round=%d 收尾",
            task_id, forced_final_after_idx, forced_round_idx,
        )
        # 进 forced final 前的"段间锁"收口：先 ensure 占位 forced final chunk，再
        # await preview/skills 收尾，最后翻 intermediateLocked。详见 _prepare_final
        # docstring。
        await _prepare_final(forced_round_idx)
        # 强制 final 路径：used = 已跑过的 0..forced_final_after_idx（含）所有 group
        # 对应的 indices 拍平。与 _v4_resolve_used_chunk_indices 的 forced 分支语义一致。
        try:
            used_indices = [
                idx
                for grp in groups_indices[: forced_final_after_idx + 1]
                for idx in grp
            ]
            used_headings = _collect_used_headings(chunks, used_indices)
            await redis_stream.set_used_headings(task_id, used_headings)
        except Exception as e:
            logger.warning(
                "[InferenceReact] task=%s 写 used_headings (forced) 失败（忽略）: %s",
                task_id, e,
            )
        verdict, full_think, full_answer = await _run_one_round(
            round_idx=forced_round_idx,
            # 沿用最后一轮中间轮的 evidence：模型刚分析完这组就判 complete,
            # 再用同一组证据 + 累积 prev_think 走 final prompt 写答案最自然。
            group_text=groups[forced_final_after_idx],
            is_last_chunk=True,
            prev_think_val=prev_think,
            step_kind="final_forced",
        )
        last_verdict = verdict or "complete"
        last_answer = full_answer
        if full_think:
            prev_think = full_think.strip()
        # forced final 收尾，翻 finalLocked 解锁【引用知识章节】段。
        await _lock_final()

    return {
        "rounds": rounds_done,
        "verdict": last_verdict,
        "answer": last_answer,
    }


# ---------------------------------------------------------------- reSearch 状态机


async def _run_research_loop(
    *,
    task_id: str,
    question: str,
    chunks: list[KnowledgeChunk],
    redis_stream: RedisStream,
    chunk_size: int,
    max_rounds: int,
    policy_id: Optional[str],
    top_n: int,
    top_m: int,
    search_fn: Optional[Callable[[str], Awaitable[list[KnowledgeChunk]]]],
    prepare_final: Callable[[int], Awaitable[None]],
    lock_final: Callable[[], Awaitable[None]],
    run_one_round: Callable[..., Awaitable[tuple[str, str, str]]],
) -> dict:
    """reSearch=true 的动作驱动状态机（与旧主循环互斥，仅在 ``run()`` 内被调用）。

    每一中间轮用 research 版 prompt 让模型产出 ``<completion>/<action>/<search-query>``：

    - ``complete``：追加一轮强制 final 写出用户答案后结束；
    - ``incomplete + paginate``：消费当前证据集的下一组（5000 字组包）；
    - ``incomplete + research``：以 ``<search-query>`` 调 :func:`hybrid_search` 召回，
      对新证据集做"单向去重"（删去已进过 ReAct 的旧块）后替换当前证据集并重置 group_idx；
      去重后为空则降级 paginate。

    所有路径共享同一 ``max_rounds`` 计数：到达预算上限时本轮强制 final 收尾；翻页耗尽
    当前证据集且仍未 complete 时也追加一轮 final 收尾。解析不可用时由
    :func:`parse_research_answer` 兜底为 ``incomplete + paginate``，不中断主流程。

    usedHeadings：累计所有"已进入 prompt"的 chunk（跨多次 research）的 heading 叶子，
    在 final 前写入快照，供 SSE 渲染【引用知识章节】段。
    """

    async def _do_search(query: str) -> list[KnowledgeChunk]:
        if search_fn is not None:
            return await search_fn(query)
        if not policy_id:
            logger.warning(
                "[InferenceReact] task=%s research 需要 policy_id 但未提供，返回空召回",
                task_id,
            )
            return []
        from .retrieval import hybrid_search

        return await hybrid_search(query, policy_id, top_n=top_n, top_m=top_m)

    current_chunks: list[KnowledgeChunk] = list(chunks or [])
    groups = pack_chunks_by_size(current_chunks, chunk_size=chunk_size)
    groups_indices = _pack_chunks_with_indices(current_chunks, chunk_size=chunk_size)
    if not groups:
        groups = ["（本次未检索到任何证据，请基于通识做最终回答。）"]
        groups_indices = []

    group_idx = 0
    round_idx = 0
    prev_think = ""
    last_answer = ""
    last_verdict = ""
    rounds_done = 0

    # 已进入过 ReAct prompt 的 chunk 去重键集合（research 单向去重的参照集）。
    seen_keys: set[str] = set()
    # 已展示 chunk 的有序集合（跨多次 research），用于 final 时聚合 usedHeadings。
    used_chunks_in_order: list[KnowledgeChunk] = []
    used_keys_for_headings: set[str] = set()

    def _mark_group_seen(gidx: int) -> None:
        if not groups_indices or gidx < 0 or gidx >= len(groups_indices):
            return
        for ci in groups_indices[gidx]:
            if 0 <= ci < len(current_chunks):
                c = current_chunks[ci]
                key = _chunk_dedup_key(c)
                seen_keys.add(key)
                if key not in used_keys_for_headings:
                    used_keys_for_headings.add(key)
                    used_chunks_in_order.append(c)

    async def _write_used_headings() -> None:
        try:
            headings: list[str] = []
            seen_h: set[str] = set()
            for c in used_chunks_in_order:
                for path in (getattr(c, "heading_paths", None) or []):
                    if not path:
                        continue
                    leaf = path[-1]
                    if isinstance(leaf, str) and leaf and leaf not in seen_h:
                        seen_h.add(leaf)
                        headings.append(leaf)
            await redis_stream.set_used_headings(task_id, headings)
        except Exception as e:
            logger.warning(
                "[InferenceReact] task=%s 写 used_headings(research) 失败（忽略）: %s",
                task_id, e,
            )

    async def _finalize(
        final_round_idx: int, group_text: str, prev_think_val: str
    ) -> tuple[str, str, str]:
        await prepare_final(final_round_idx)
        await _write_used_headings()
        verdict, _think, answer = await run_one_round(
            round_idx=final_round_idx,
            group_text=group_text,
            is_last_chunk=True,
            prev_think_val=prev_think_val,
            step_kind="final",
        )
        await lock_final()
        return verdict, _think, answer

    _placeholder = "（本次未检索到任何证据，请基于通识做最终回答。）"

    while True:
        rounds_done = round_idx + 1
        budget_last = max_rounds > 0 and round_idx >= max_rounds - 1
        group_text = groups[group_idx] if group_idx < len(groups) else _placeholder
        if group_idx < len(groups):
            _mark_group_seen(group_idx)

        # 预算用尽：本轮强制 final 收尾。
        if budget_last:
            verdict, _think, answer = await _finalize(round_idx, group_text, prev_think)
            last_verdict = verdict or "complete"
            last_answer = answer
            break

        # 中间轮（research 协议）。
        verdict, full_think, full_answer = await run_one_round(
            round_idx=round_idx,
            group_text=group_text,
            is_last_chunk=False,
            prev_think_val=prev_think,
            step_kind="intermediate",
            research_mode=True,
        )
        if full_think:
            prev_think = full_think.strip()

        decision = parse_research_answer(full_answer)
        completion = decision["completion"]
        action = decision["action"]
        search_query = decision["search_query"]
        logger.info(
            "[InferenceReact] task=%s round=%d research 决策 completion=%s action=%s query=%r",
            task_id, round_idx, completion, action, search_query,
        )

        # complete：追加一轮强制 final 写答案。
        if completion == "complete":
            forced_round_idx = round_idx + 1
            rounds_done = forced_round_idx + 1
            verdict, _think, answer = await _finalize(
                forced_round_idx, group_text, prev_think
            )
            last_verdict = verdict or "complete"
            last_answer = answer
            break

        # incomplete：先尝试 research，失败/为空则降级 paginate。
        advanced = False
        if action == "research" and search_query:
            try:
                new_chunks = await _do_search(search_query)
            except Exception as e:
                logger.warning(
                    "[InferenceReact] task=%s research 检索异常，降级 paginate: %s",
                    task_id, e,
                )
                new_chunks = []
            # 单向删减：从新证据集中删去"已进过 ReAct 的旧块"，老证据集不改。
            filtered = [
                c for c in (new_chunks or [])
                if _chunk_dedup_key(c) not in seen_keys
            ]
            logger.info(
                "[InferenceReact] task=%s round=%d research 召回 %d，单向去重后 %d",
                task_id, round_idx, len(new_chunks or []), len(filtered),
            )
            if filtered:
                current_chunks = filtered
                groups = pack_chunks_by_size(current_chunks, chunk_size=chunk_size)
                groups_indices = _pack_chunks_with_indices(
                    current_chunks, chunk_size=chunk_size
                )
                group_idx = 0
                advanced = True

        if not advanced:
            # paginate：消费下一组；若已无后续证据组，则追加一轮 final 收尾。
            group_idx += 1
            if group_idx >= len(groups):
                forced_round_idx = round_idx + 1
                rounds_done = forced_round_idx + 1
                verdict, _think, answer = await _finalize(
                    forced_round_idx, group_text, prev_think
                )
                last_verdict = verdict or "complete"
                last_answer = answer
                break

        round_idx += 1

    return {
        "rounds": rounds_done,
        "verdict": last_verdict,
        "answer": last_answer,
    }

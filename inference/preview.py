"""Step1：preview 阶段。

- 用 :func:`inference.llm_stream.chat_stream` 拿到 think+answer 双通道流；
- 通过 ``asyncio.Queue`` + 独立 flush 任务做 TPS 节流，避免 redis 被高频小写打爆；
- 完成后置 ``preview.done=True``。

节流策略：每 ``1/PREVIEW_TPS`` 秒批量 flush 一次累积的 think/answer 增量到 redis。
这样接口侧轮询 redis 看到的视效，自然就是 ``PREVIEW_TPS`` 字次/秒级别的递增。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from utils.verbose_logger import (
    is_session_active,
    log_llm_call,
    log_llm_error,
    step_scope,
)

from . import config
from .llm_stream import StreamTagRouter, chat_stream
from .redis_stream import RedisStream

logger = logging.getLogger(__name__)


class _ChannelBuffer:
    """think / answer 各自的累积缓冲，flush 时整段 append 到 redis。"""

    __slots__ = ("think", "answer")

    def __init__(self) -> None:
        self.think: list[str] = []
        self.answer: list[str] = []

    def push(self, channel: str, delta: str) -> None:
        if not delta:
            return
        if channel == "think":
            self.think.append(delta)
        elif channel == "answer":
            self.answer.append(delta)

    def take(self) -> tuple[str, str]:
        t = "".join(self.think)
        a = "".join(self.answer)
        self.think.clear()
        self.answer.clear()
        return t, a

    def empty(self) -> bool:
        return not self.think and not self.answer


async def _flush_loop(
    task_id: str,
    redis_stream: RedisStream,
    buffer: _ChannelBuffer,
    interval: float,
    stop: asyncio.Event,
) -> None:
    """周期性把 buffer 里的累积内容 flush 到 redis。"""

    try:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
            think_delta, answer_delta = buffer.take()
            if think_delta:
                await redis_stream.append_preview(task_id, "think", think_delta)
            if answer_delta:
                await redis_stream.append_preview(task_id, "answer", answer_delta)
    except asyncio.CancelledError:
        pass
    finally:
        # 退出前再 flush 一次剩余增量。
        think_delta, answer_delta = buffer.take()
        if think_delta:
            await redis_stream.append_preview(task_id, "think", think_delta)
        if answer_delta:
            await redis_stream.append_preview(task_id, "answer", answer_delta)


async def run(
    task_id: str,
    question: str,
    redis_stream: RedisStream,
    *,
    vendor: str = "servyou",
    model: str = "deepseek-v3.2-1163259bcc6c",
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    topic_general_knowledge: Optional[str] = None,
    policy_id: Optional[str] = None,
    case_policy_ids: Optional[list[str]] = None,
    case_top_k: int = 0,
    case_sim_threshold: float = 0.85,
    tps: int = config.PREVIEW_TPS,
) -> None:
    """运行 preview 阶段。

    ``topic_general_knowledge``：用于 :func:`inference.prompts.select_preview_prompt`
    的路由——非空时走 ``PREVIEW_*_WITH_TGK`` 双件套（system 要求"遵循专题通用知识"，
    user prompt 注入【专题通用知识】小节）；为空/None 时回落到老 ``PREVIEW_*``
    （只有【用户问题】、system 仅要求"基于自身常识"），与改造前完全等价。

    ``case_top_k`` / ``case_sim_threshold`` / ``policy_id``：preview 阶段 case 库
    召回参数。``case_top_k=0`` 或 ``policy_id`` 为空时**完全跳过 case 检索**,
    走原 2 套 PREVIEW_* prompt；``case_top_k>0`` 时调
    :func:`inference.retrieval.case_search.search_cases`，按 cosine_similarity
    阈值过滤后取 top-k，命中非空则走带【历史经验】的新 prompt。case 检索
    全链路异常被吃掉返回 ``[]``，preview 不会因此卡死。

    ``case_policy_ids``：专题定位返回**多个**候选专题时的完整 policyId 列表。
    非空且包含多个专题时，case 检索走
    :func:`inference.retrieval.case_search.search_cases_multi`，对所有专题各自
    ``case_{khCode}`` collection **并发**独立召回后合并去重（每专题保留分桶 + 阈值 +
    topC 逻辑）；为空/单专题时回落到单集合 ``search_cases``（用 ``policy_id``）。

    显式传 ``system_prompt`` / ``user_prompt`` 时优先使用对应入参，路由结果被覆盖
    （此时 case 检索仍会触发，但其结果不会影响 prompt——保留触发是为了让上层
    需要日志/统计时仍能复用本入口；如需彻底跳过，传 ``case_top_k=0``）。

    异常会被捕获并写入 ``preview.done=True`` + 日志，不向上抛，确保 pipeline
    的 react 主路径不会被 preview 拖死。
    """

    from .prompts import select_preview_prompt

    related_cases: list = []
    # 多专题命中（topic locator 多候选）→ fan-out 到所有专题 collection 并发检索；
    # 否则走单集合 search_cases。用 policy_id 兜底保证至少能查首专题。
    multi_policy_ids = [p for p in (case_policy_ids or []) if p]
    if case_top_k > 0 and (policy_id or multi_policy_ids):
        try:
            if len(multi_policy_ids) > 1:
                from .retrieval.case_search import search_cases_multi

                related_cases = await search_cases_multi(
                    question, multi_policy_ids,
                    threshold=case_sim_threshold,
                    top_k=case_top_k,
                )
            else:
                from .retrieval.case_search import search_cases

                related_cases = await search_cases(
                    question, policy_id or multi_policy_ids[0],
                    threshold=case_sim_threshold,
                    top_k=case_top_k,
                )
        except Exception as e:
            # search_cases 内部已经全兜底；这里双保险，确保任何意外都不阻塞 preview。
            logger.warning(
                "[InferencePreview] task=%s case_search 失败（忽略）: %s",
                task_id, e,
            )
            related_cases = []

    default_sys_p, default_usr_p = select_preview_prompt(
        question=question,
        topic_general_knowledge=topic_general_knowledge,
        related_cases=related_cases,
    )
    sys_p = system_prompt or default_sys_p
    usr_p = user_prompt or default_usr_p

    interval = 1.0 / max(int(tps), 1)
    buffer = _ChannelBuffer()
    stop = asyncio.Event()
    flush_task = asyncio.create_task(
        _flush_loop(task_id, redis_stream, buffer, interval, stop),
        name=f"preview-flush:{task_id}",
    )

    # 标签状态机：vendor 把 think 直接塞 content 时由它二次拆分。
    router = StreamTagRouter()

    def _on_think(s: str) -> None:
        buffer.push("think", s)

    def _on_answer(s: str) -> None:
        buffer.push("answer", s)

    # verbose 模式下整段累积 prompt/response，待 stream 结束统一写一次 log_llm_call,
    # 与 llm.client.chat 的"一次性日志条目"语义保持一致。
    verbose_on = is_session_active()
    collected_think: list[str] = []
    collected_answer: list[str] = []
    t0 = time.time() if verbose_on else None

    def _on_think_collect(s: str) -> None:
        if verbose_on and s:
            collected_think.append(s)
        _on_think(s)

    def _on_answer_collect(s: str) -> None:
        if verbose_on and s:
            collected_answer.append(s)
        _on_answer(s)

    stream_error: Exception | None = None
    try:
        with step_scope("inference_preview", prompt_vars={"question": question}):
            try:
                async for channel, delta in chat_stream(
                    usr_p,
                    vendor=vendor,
                    model=model,
                    system=sys_p,
                    enable_thinking=True,
                ):
                    if channel == "think":
                        if verbose_on and delta:
                            collected_think.append(delta)
                        buffer.push("think", delta)
                    else:
                        # answer 通道的内容里可能再夹 <think>/<answer> 标签，过 router
                        router.feed(delta, on_think=_on_think_collect, on_answer=_on_answer_collect)
            except Exception as e:
                stream_error = e
                logger.exception("[InferencePreview] task=%s 失败: %s", task_id, e)
            finally:
                if verbose_on:
                    think_text = "".join(collected_think).strip()
                    answer_text = "".join(collected_answer).strip()
                    response_text = (
                        f"<think>\n{think_text}\n</think>\n{answer_text}"
                        if think_text else answer_text
                    )
                    elapsed_ms = int((time.time() - t0) * 1000) if t0 is not None else None
                    try:
                        if stream_error is None:
                            log_llm_call(
                                prompt=usr_p,
                                response=response_text,
                                system=sys_p,
                                vendor=vendor,
                                model=model,
                                elapsed_ms=elapsed_ms,
                                extra={
                                    "stage": "inference_preview",
                                    "task_id": task_id,
                                    "enable_thinking": True,
                                },
                            )
                        else:
                            log_llm_error(
                                prompt=usr_p,
                                error=f"{type(stream_error).__name__}: {stream_error}",
                                system=sys_p,
                                vendor=vendor,
                                model=model,
                                elapsed_ms=elapsed_ms,
                                extra={
                                    "stage": "inference_preview",
                                    "task_id": task_id,
                                    "partial_think": think_text[:500],
                                    "partial_answer": answer_text[:500],
                                },
                            )
                    except Exception as log_e:
                        logger.debug(
                            "[InferencePreview] verbose 落盘失败（忽略）: %s", log_e
                        )
    finally:
        stop.set()
        try:
            await flush_task
        except Exception as e:
            logger.warning("[InferencePreview] task=%s flush_task 退出异常: %s", task_id, e)
        await redis_stream.set_preview_done(task_id, True)

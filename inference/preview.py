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
    vendor: str = "aliyun",
    model: str = "deepseek-v3.2",
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    tps: int = config.PREVIEW_TPS,
) -> None:
    """运行 preview 阶段。

    异常会被捕获并写入 ``preview.done=True`` + 日志，不向上抛，确保 pipeline
    的 react 主路径不会被 preview 拖死。
    """

    from .prompts import PREVIEW_SYSTEM_PROMPT, PREVIEW_USER_PROMPT

    sys_p = system_prompt or PREVIEW_SYSTEM_PROMPT
    usr_p = user_prompt or PREVIEW_USER_PROMPT.format(question=question)

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

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
import logging
import time
from typing import Optional

from reasoner.v3.chunk_builder import KnowledgeChunk
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
) -> tuple[str, str, str]:
    """跑一轮，返回 ``(verdict, full_think, full_answer)``。

    - ``store_think_to_chunk=False`` 时，本轮模型产出的 think 仅累积在内存中（用于
      下一轮的 ``prev_think``），不写入 ``react.chunks[round_idx].think``，对应默认
      模式下中间轮 ``chunk.think=""`` 的约定。
    - ``answer`` 始终写到 ``chunk.answer``（聚合时是否进接口 think/answer 由开关决定）。
    """

    write_queue: asyncio.Queue[Optional[tuple[str, str]]] = asyncio.Queue()

    async def writer_loop() -> None:
        while True:
            item = await write_queue.get()
            if item is None:
                return
            channel, delta = item
            try:
                await redis_stream.append_react_chunk_delta(
                    task_id, round_idx, channel, delta, is_last_chunk=is_last_chunk
                )
            except Exception as e:
                # 一笔失败不影响后续：下游会从 redis 丢失对应增量，但不会卡死流。
                logger.warning(
                    "[InferenceReact] task=%s round=%d 写 redis 失败 ch=%s: %s",
                    task_id, round_idx, channel, e,
                )

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
            write_queue.put_nowait(("think", s))

    def _emit_answer(s: str) -> None:
        if not s:
            return
        full_answer_parts.append(s)
        write_queue.put_nowait(("answer", s))

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
        await write_queue.put(None)
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

    verdict = (router.verdict or "").strip().lower()
    full_think = "".join(full_think_parts).strip()
    full_answer = "".join(full_answer_parts).strip()
    return verdict, full_think, full_answer


# ---------------------------------------------------------------- 主入口


async def run(
    task_id: str,
    question: str,
    chunks: list[KnowledgeChunk],
    redis_stream: RedisStream,
    *,
    vendor: str = "aliyun",
    model: str = "deepseek-v3.2",
    chunk_size: int = config.CHUNK_SIZE,
    max_rounds: int = config.REACT_MAX_ROUNDS,
    intermediate_think_enabled: Optional[bool] = None,
) -> dict:
    """运行 ReAct 主循环。返回 ``{"rounds": int, "verdict": str, "answer": str}``。

    - 任意一轮 ``verdict == "complete"`` 即结束；
    - 否则跑到最后一组证据（受 ``max_rounds`` 限制），强制按"最终轮"出 answer；
    - 入参 ``chunks`` 为空时直接以一轮 "无检索证据" 跑最终轮。
    """

    flag = (
        bool(intermediate_think_enabled)
        if intermediate_think_enabled is not None
        else bool(config.REACT_INTERMEDIATE_THINK_ENABLED)
    )

    groups = pack_chunks_by_size(chunks, chunk_size=chunk_size)
    if not groups:
        groups = ["（本次未检索到任何证据，请基于通识做最终回答。）"]
    if max_rounds > 0 and len(groups) > max_rounds:
        # 超额时把尾部多余 group 合并为最后一组，确保至少跑一次"最终轮"。
        head = groups[: max_rounds - 1]
        tail = _CHUNK_SEPARATOR.join(groups[max_rounds - 1 :])
        groups = head + [tail]

    await redis_stream.set_status(task_id, "reasoning")

    # 仅保留“上一轮”的思考摘要，避免跨轮无限膨胀。
    prev_think = ""
    last_verdict = ""
    last_answer = ""
    total_groups = len(groups)
    rounds_done = 0

    for round_idx, group_text in enumerate(groups):
        rounds_done = round_idx + 1
        is_last_chunk = round_idx == total_groups - 1
        # 提前占位，确保 chunk[round_idx] 存在，方便下游（SSE）观测进度
        await redis_stream.ensure_react_chunk(task_id, round_idx)

        snapshot = await redis_stream.get(task_id) or {}
        preview_snapshot = snapshot.get("preview")
        skills_snapshot = snapshot.get("skills")

        system_prompt, user_prompt = select_react_prompt(
            is_last_chunk=is_last_chunk,
            question=question,
            evidence=group_text,
            prev_think=prev_think,
            preview=preview_snapshot,
            skills=skills_snapshot,
        )

        enable_thinking = is_last_chunk or flag
        store_think = is_last_chunk or flag

        step_label = (
            f"inference_react_final_r{round_idx}"
            if is_last_chunk else f"inference_react_intermediate_r{round_idx}"
        )
        try:
            with step_scope(step_label, prompt_vars={
                "round_idx": round_idx,
                "is_last_chunk": is_last_chunk,
                "total_groups": total_groups,
            }):
                verdict, full_think, full_answer = await _stream_react_round(
                    task_id,
                    round_idx,
                    redis_stream,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    vendor=vendor,
                    model=model,
                    is_last_chunk=is_last_chunk,
                    enable_thinking=enable_thinking,
                    store_think_to_chunk=store_think,
                )
        except Exception as e:
            logger.exception(
                "[InferenceReact] task=%s round=%d 模型流式失败: %s", task_id, round_idx, e
            )
            await redis_stream.set_react_chunk_complete(
                task_id, round_idx, complete=False, verdict="error"
            )
            raise

        await redis_stream.set_react_chunk_complete(
            task_id, round_idx, complete=True, verdict=verdict or ("complete" if is_last_chunk else "incomplete"),
        )

        last_verdict = verdict
        last_answer = full_answer
        if full_think:
            # 下一轮只继承上一轮 think（由模型在本轮中自行完成降噪提炼），
            # 不再累积所有历史轮次，避免 prompt 体积持续膨胀。
            prev_think = full_think.strip()

        if verdict == "complete" or is_last_chunk:
            break

    return {
        "rounds": rounds_done,
        "verdict": last_verdict,
        "answer": last_answer,
    }

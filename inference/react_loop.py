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
) -> tuple[str, str, str]:
    """跑一轮，返回 ``(verdict, full_think, full_answer)``。

    - ``store_think_to_chunk=False`` 时，本轮模型产出的 think 仅累积在内存中（用于
      下一轮的 ``prev_think``），不写入 ``react.chunks[round_idx].think``，对应默认
      模式下中间轮 ``chunk.think=""`` 的约定。
    - ``answer_as_verdict=False``（最终轮）：``<answer>`` 是真正的用户答案，按增量
      写到 ``chunk.answer``，``verdict`` 取自 ``router.verdict``（旧 <verdict> 标签）。
    - ``answer_as_verdict=True``（中间轮，新 prompt 协议）：``<answer>`` 内容固定是
      ``complete`` / ``incomplete`` 的裁决字符串，仅作 verdict 使用，**不** 写到
      ``chunk.answer``（否则会被 SSE 聚合误当成真实答案显示）；``verdict`` 直接取自
      ``full_answer.strip().lower()``。
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
        # 中间轮 <answer> 内容只是 complete/incomplete 的 verdict 字符串，
        # 不能写到 chunk.answer——否则 SSE 聚合会把它当成最终答案吐给前端。
        if not answer_as_verdict:
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

    full_think = "".join(full_think_parts).strip()
    full_answer = "".join(full_answer_parts).strip()
    if answer_as_verdict:
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
    vendor: str = "aliyun",
    model: str = "deepseek-v3.2",
    chunk_size: int = config.CHUNK_SIZE,
    max_rounds: int = config.REACT_MAX_ROUNDS,
    intermediate_think_enabled: Optional[bool] = None,
) -> dict:
    """运行 ReAct 主循环。返回 ``{"rounds": int, "verdict": str, "answer": str}``。

    流程：
    - 按 ``groups`` 顺序逐轮跑中间轮（``answer_as_verdict=True``）：
      ``<think>`` 累进 ``prev_think``，``<answer>`` 内容是 ``complete``/``incomplete`` 裁决。
    - 任意一轮裁决 ``complete``：立刻追加**一轮强制 final**（``is_last_chunk=True``），
      用同一组证据 + 累积 ``prev_think`` 让模型真正写出用户可见答案，然后退出。
    - 跑到最后一组证据仍未 ``complete``：最后一组本身就按 final 跑，正常收尾。
    - 入参 ``chunks`` 为空时直接以一轮 "无检索证据" 跑最终轮。
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

    # 仅保留"上一轮"的思考摘要，避免跨轮无限膨胀。
    prev_think = ""
    last_verdict = ""
    last_answer = ""
    total_groups = len(groups)
    rounds_done = 0

    async def _run_one_round(
        *, round_idx: int, group_text: str, is_last_chunk: bool, prev_think_val: str,
        step_kind: str,
    ) -> tuple[str, str, str]:
        """跑单轮：占位 chunk、读快照、组装 prompt、流式调模型、收尾标记。

        ``step_kind`` 用于 verbose 日志的 step label 后缀（``intermediate`` / ``final`` /
        ``final_forced``），便于在 jsonl 中区分"模型自己跑到最后一组" 与"中间判 complete
        后强制追加 final" 两类最终轮。
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
                    # 中间轮的 <answer> 是 verdict 字符串；最终轮的 <answer> 才是用户答案。
                    answer_as_verdict=not is_last_chunk,
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

    # ---- 主循环：依次跑各组证据 -----------------------------------
    forced_final_after_idx: int | None = None
    for round_idx, group_text in enumerate(groups):
        rounds_done = round_idx + 1
        is_last_chunk = round_idx == total_groups - 1
        # 正常收尾分支：本轮就是 final（最后一组），在 _run_one_round 调用前
        # 先把"全部进入过 prompt 的 chunks"对应的 heading 叶子写到 redis 快照,
        # 让 SSE think 第一帧 final.think delta 落地的同时就带上【引用知识章节】段。
        # 注意这里 used = 全部 indices：模型跑到最后一组自然收尾时所有 group 都被消费。
        if is_last_chunk:
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

    return {
        "rounds": rounds_done,
        "verdict": last_verdict,
        "answer": last_answer,
    }

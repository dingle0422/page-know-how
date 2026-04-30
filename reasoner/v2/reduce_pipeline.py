"""通用 Reduce Pipeline：所有 batch summary 共享一个 "凑批 + 回灌" 流水线。

设计动机
--------
原 layered 实现把多次压缩组织成"层"：本层 N 个 batch 必须 as_completed 等齐才能进入
下一层。这在层数深、batch 数多、且各 batch 耗时长尾差异大时会引入不必要的等待。

ReducePipeline 把"层"的概念抹掉：
  - 所有产出（chunk LLM、派生 LLM、子智能体 result、召回 fragment、上一次 BATCH_SUMMARY
    的输出）都进同一个 pending_pool；
  - 队列长度凑够 batch_size 立即从队头 popleft 一批，提交到 bs_pool 跑 BATCH_SUMMARY；
  - BATCH_SUMMARY 的输出作为新 ReducePart（depth + 1）回灌入队；
  - 所有上游生产者（含中间 BATCH_SUMMARY 自身）全部完成、且队内 part 数 ≤ batch_size 时
    收口，触发调用方提供的 final_merge_callable。

护栏
----
单 part 经过的压缩次数有上限（max_part_depth）。命中上限的 part 不再参与凑批，转入
frozen 列表，直接保留到 final merge。避免某些 part 被反复压缩造成信息严重损耗。

线程安全
--------
所有 pending_pool / active_producers / frozen 的读写都在同一个 Condition 锁内完成。
producer 由调用方驱动（producer_inc / submit_part / producer_done），中间 batch
本身的 in-flight 计数由 pipeline 内部维护。
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from llm.client import chat
from utils.verbose_logger import step_scope
from reasoner.v2.prompts import BATCH_REDUCE_SYSTEM_PROMPT, BATCH_SUMMARY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class ReducePart:
    """流水线内的一个数据单元。

    - text: part 的原文（中间产出 / 派生 chunk LLM 输出 / batch summary 输出皆同构）
    - source_label: 用于 trace 与日志的简短标签（如 "chunk-3" / "derived-3.2" /
      "agent-5" / "frag-12" / "summary-d1-b2"）
    - depth: 经过的 BATCH_SUMMARY 次数；原始产出 0，每被压缩一次 +1
    """
    text: str
    source_label: str
    depth: int = 0


@dataclass
class _BatchTrace:
    """一次中间 batch 的轨迹记录，用于事后渲染流水线轨迹章节。"""
    batch_seq: int
    depth_in: int                    # 输入 part 的最大 depth
    depth_out: int                   # 输出 part 的 depth (= depth_in + 1)
    input_labels: list[str]
    output_label: str
    prompt_chars: int
    output_chars: int
    elapsed_ms: int
    failed: bool = False
    error: str = ""


class ReducePipeline:
    """通用 reduce 流水线：凑批 + 回灌 + 收口 final merge。

    使用模式：
        pipe = ReducePipeline(
            batch_size=3,
            intermediate_prompt=BATCH_SUMMARY_PROMPT,
            final_merge_callable=lambda parts: my_final_merge(parts, layer=...),
            question=question, vendor=vendor, model=model,
        )
        pipe.start()
        for chunk in chunks:
            pipe.producer_inc(1)
            executor.submit(my_worker, chunk, pipe)   # worker 内 submit_part / producer_done
        answer = pipe.wait_and_finalize()
    """

    def __init__(
        self,
        *,
        batch_size: int,
        intermediate_prompt: str,
        final_merge_callable: Callable[[list[ReducePart]], str],
        question: str,
        vendor: str,
        model: str,
        max_part_depth: int = 5,
        bs_workers: int = 5,
        intermediate_system_prompt: str = BATCH_REDUCE_SYSTEM_PROMPT + "\n\n" + BATCH_SUMMARY_SYSTEM_PROMPT,
        logger_label: str = "ReduceQueue",
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须 > 0，收到 {batch_size}")
        if max_part_depth <= 0:
            raise ValueError(f"max_part_depth 必须 > 0，收到 {max_part_depth}")

        self.batch_size = int(batch_size)
        self.intermediate_prompt = intermediate_prompt
        self.intermediate_system_prompt = intermediate_system_prompt
        self.final_merge_callable = final_merge_callable
        self.question = question
        self.vendor = vendor
        self.model = model
        self.max_part_depth = int(max_part_depth)
        self.logger_label = logger_label

        self._bs_pool = ThreadPoolExecutor(
            max_workers=max(1, int(bs_workers)),
            thread_name_prefix="reduce-bs",
        )
        self._owns_pool = True

        self._cond = threading.Condition()
        self._pending: deque[ReducePart] = deque()
        self._frozen: list[ReducePart] = []      # 已达 max_part_depth，不再压缩
        self._active_producers: int = 0          # 调用方维护
        self._inflight_batches: int = 0          # pipeline 内部维护
        self._closed: bool = False

        self._batch_seq: int = 0
        self._traces: list[_BatchTrace] = []
        self._traces_lock = threading.Lock()

    # ------------------- 生产者 API -------------------

    def producer_inc(self, n: int = 1) -> None:
        """登记 n 个未来会调 producer_done 的生产者任务。"""
        if n <= 0:
            return
        with self._cond:
            self._active_producers += n

    def producer_done(self, n: int = 1) -> None:
        """归还 n 个生产者任务的完成信号。可能触发收口。"""
        if n <= 0:
            return
        with self._cond:
            self._active_producers -= n
            if self._active_producers < 0:
                logger.error(
                    f"[{self.logger_label}] active_producers < 0，存在 producer_inc/done "
                    f"配对错误。当前值: {self._active_producers}"
                )
                self._active_producers = 0
            self._cond.notify_all()

    def submit_part(self, part: ReducePart) -> None:
        """投递一个 part 到流水线。如果 depth 已达上限，直接进 frozen 列表。
        否则入 pending_pool，并在持锁状态下检查是否凑齐一批触发 flush。
        """
        with self._cond:
            if part.depth >= self.max_part_depth:
                self._frozen.append(part)
                logger.info(
                    f"[{self.logger_label}] part {part.source_label} 达到最大压缩深度 "
                    f"({part.depth}/{self.max_part_depth})，进入 frozen 列表"
                )
            else:
                self._pending.append(part)
            self._maybe_flush_locked()
            self._cond.notify_all()

    # ------------------- 终止与等待 -------------------

    def wait_and_finalize(self) -> str:
        """阻塞直到所有上游 + 中间 batch 完成，然后调 final_merge_callable 返回 answer。"""
        with self._cond:
            while not self._is_settled_locked():
                self._cond.wait()
            final_parts = list(self._pending) + list(self._frozen)
            self._pending.clear()
            self._frozen.clear()
            self._closed = True

        logger.info(
            f"[{self.logger_label}] 收口：剩余 pending+frozen 共 {len(final_parts)} 条 → final merge"
        )
        for p in final_parts:
            logger.info(
                f"[{self.logger_label}]   - {p.source_label} (depth={p.depth}, {len(p.text)} 字符)"
            )

        try:
            answer = self.final_merge_callable(final_parts)
        finally:
            self._shutdown()
        return answer

    def get_traces(self) -> list[_BatchTrace]:
        """返回所有中间 batch 的轨迹快照（按完成顺序）。"""
        with self._traces_lock:
            return list(self._traces)

    def render_trace_section(self) -> str:
        """渲染一段 trace_log 末尾追加用的流水线轨迹文本。"""
        traces = self.get_traces()
        lines: list[str] = [f"=== [{self.logger_label}] 流水线轨迹 ==="]
        if not traces:
            lines.append("（未触发任何中间 BATCH_SUMMARY，所有 part 直接进入 final merge）")
            return "\n".join(lines)
        lines.append(
            f"共触发 {len(traces)} 次中间 BATCH_SUMMARY"
            f"（max_part_depth={self.max_part_depth}, batch_size={self.batch_size}）"
        )
        for t in traces:
            status = "FAILED" if t.failed else "ok"
            lines.append(
                f"  - batch#{t.batch_seq} depth {t.depth_in}->{t.depth_out} "
                f"({len(t.input_labels)} parts, prompt={t.prompt_chars}c, "
                f"out={t.output_chars}c, {t.elapsed_ms}ms) [{status}]"
            )
            for lbl in t.input_labels:
                lines.append(f"        in : {lbl}")
            lines.append(f"        out: {t.output_label}")
            if t.failed and t.error:
                lines.append(f"        err: {t.error[:200]}")
        return "\n".join(lines)

    # ------------------- 内部实现 -------------------

    def _is_settled_locked(self) -> bool:
        """收口判定（必须在持有 self._cond 的情况下调用）。"""
        if self._active_producers > 0 or self._inflight_batches > 0:
            return False
        # 上游全完，无中间 batch 在跑：剩余 part 数 ≤ batch_size 即可收口；
        # 若仍 > batch_size，说明还能再压一批
        return len(self._pending) <= self.batch_size

    def _maybe_flush_locked(self) -> None:
        """持锁时检查并 flush 所有可触发的 batch。

        注意：本方法只调度 bs_pool.submit；done callback 内部会再次拿锁触发回灌检查。
        """
        while len(self._pending) >= self.batch_size:
            batch_parts = [self._pending.popleft() for _ in range(self.batch_size)]
            self._batch_seq += 1
            batch_seq = self._batch_seq
            self._inflight_batches += 1
            depth_in = max(p.depth for p in batch_parts)
            depth_out = depth_in + 1
            input_labels = [p.source_label for p in batch_parts]

            try:
                fut = self._bs_pool.submit(
                    self._run_intermediate_batch,
                    batch_seq, batch_parts, depth_in, depth_out,
                )
            except RuntimeError as e:
                # 池已 shutdown 等异常：还原计数并把 part 退回 pending
                self._inflight_batches -= 1
                for p in reversed(batch_parts):
                    self._pending.appendleft(p)
                logger.error(
                    f"[{self.logger_label}] bs_pool.submit 失败，回滚 batch#{batch_seq}: {e}"
                )
                break

            fut.add_done_callback(
                lambda f, seq=batch_seq, lbls=input_labels, din=depth_in, dout=depth_out:
                self._on_batch_done(f, seq, lbls, din, dout)
            )

    def _run_intermediate_batch(
        self,
        batch_seq: int,
        batch_parts: list[ReducePart],
        depth_in: int,
        depth_out: int,
    ) -> tuple[str, int, int]:
        """worker：跑一次 BATCH_SUMMARY，返回 (output_text, prompt_chars, elapsed_ms)。"""
        import time
        batch_content = "\n\n".join(p.text for p in batch_parts)
        prompt = self.intermediate_prompt.format(
            batch_index=batch_seq,
            total_batches="?",
            question=self.question,
            batch_content=batch_content,
        )
        prompt_chars = len(prompt)
        logger.info(
            f"[{self.logger_label}] batch#{batch_seq} depth {depth_in}->{depth_out} "
            f"({len(batch_parts)} parts) prompt={prompt_chars} 字符"
        )
        t0 = time.time()
        try:
            with step_scope(f"reduce_intermediate·b{batch_seq}·d{depth_in}->d{depth_out}"):
                output = chat(
                    prompt, vendor=self.vendor, model=self.model,
                    system=self.intermediate_system_prompt,
                )
        except Exception as e:
            elapsed = int((time.time() - t0) * 1000)
            logger.error(f"[{self.logger_label}] batch#{batch_seq} 失败: {e}")
            # 失败时把原始 batch_content 作为输出，避免信息丢失（但 depth 仍 +1，
            # 防止反复重压同一段内容陷入死循环）
            fallback = (
                f"（batch#{batch_seq} 中间压缩失败: {e}）\n"
                f"原始内容:\n{batch_content}"
            )
            raise _BatchFailure(fallback, prompt_chars, elapsed, str(e)) from e
        elapsed = int((time.time() - t0) * 1000)
        return output, prompt_chars, elapsed

    def _on_batch_done(
        self,
        fut: Future,
        batch_seq: int,
        input_labels: list[str],
        depth_in: int,
        depth_out: int,
    ) -> None:
        """中间 batch 完成回调：把输出当 ReducePart 回灌、归还 in-flight 计数。"""
        failed = False
        err = ""
        try:
            output, prompt_chars, elapsed_ms = fut.result()
            output_chars = len(output)
        except _BatchFailure as bf:
            output, prompt_chars, elapsed_ms = bf.fallback, bf.prompt_chars, bf.elapsed_ms
            output_chars = len(output)
            failed = True
            err = bf.error
        except Exception as e:
            # 未知异常的最后兜底
            output = f"（batch#{batch_seq} 中间压缩异常: {e}）"
            output_chars = len(output)
            prompt_chars = 0
            elapsed_ms = 0
            failed = True
            err = str(e)

        output_label = f"summary-b{batch_seq}-d{depth_out}"
        with self._traces_lock:
            self._traces.append(_BatchTrace(
                batch_seq=batch_seq,
                depth_in=depth_in,
                depth_out=depth_out,
                input_labels=list(input_labels),
                output_label=output_label,
                prompt_chars=prompt_chars,
                output_chars=output_chars,
                elapsed_ms=elapsed_ms,
                failed=failed,
                error=err,
            ))

        new_part = ReducePart(text=output, source_label=output_label, depth=depth_out)

        with self._cond:
            self._inflight_batches -= 1
            if new_part.depth >= self.max_part_depth:
                self._frozen.append(new_part)
                logger.info(
                    f"[{self.logger_label}] batch#{batch_seq} 输出达到最大压缩深度，"
                    f"进入 frozen"
                )
            else:
                self._pending.append(new_part)
            self._maybe_flush_locked()
            self._cond.notify_all()

    def _shutdown(self) -> None:
        if self._owns_pool:
            try:
                self._bs_pool.shutdown(wait=True)
            except Exception:
                pass


class _BatchFailure(Exception):
    """中间 batch 失败时携带 fallback 文本与统计的内部异常，仅在 _on_batch_done 内被捕获。"""
    def __init__(self, fallback: str, prompt_chars: int, elapsed_ms: int, error: str):
        super().__init__(error)
        self.fallback = fallback
        self.prompt_chars = prompt_chars
        self.elapsed_ms = elapsed_ms
        self.error = error

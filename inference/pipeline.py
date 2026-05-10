"""inference 主调度：preview / skills / hybrid 检索 / ReAct 多轮推理。

设计：

- preview / skills / 检索三件并发跑（``asyncio.create_task``）；
- 检索结果回来后立即开 react_loop；
- preview / skills 即便晚于 react_loop 启动也无所谓，react_loop 在每轮内会重读
  redis 里的最新 preview / skills 快照；
- 任意阶段异常都被翻译成 ``status="failed"`` + ``error="..."`` 写回 redis；
- 终态 ``status in {"done", "failed"}`` 后调用方（SSE relay）会再 push 一帧并断流。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from .react_loop import run as run_react_loop
from .redis_stream import RedisStream
from .retrieval import hybrid_search

logger = logging.getLogger(__name__)


@dataclass
class InferenceOptions:
    vendor: str = "qwen3.5-122b-a10b"
    model: str = "Qwen3.5-122B-A10B"
    preview_enabled: bool = True
    skills_enabled: bool = True
    top_n: int = 20
    top_m: int = 20
    intermediate_think_enabled: Optional[bool] = None  # None=用全局开关


async def _await_or_log(coro, *, label: str) -> None:
    """跑后台任务（preview/skills），异常仅打日志，不影响主流程。"""

    try:
        await coro
    except Exception as e:
        logger.exception("[InferencePipeline] %s 失败: %s", label, e)


async def run(
    task_id: str,
    question: str,
    policy_id: str,
    redis_stream: RedisStream,
    *,
    options: Optional[InferenceOptions] = None,
) -> None:
    """主调度。本协程跑完即代表整个 inference 任务完成（或失败）。

    异常被自吞 + 写 ``status=failed``，确保 SSE relay 一定能拿到终态。
    """

    opts = options or InferenceOptions()
    bg_tasks: list[asyncio.Task] = []

    try:
        await redis_stream.set_status(task_id, "preview")

        if opts.preview_enabled:
            from .preview import run as run_preview

            bg_tasks.append(asyncio.create_task(
                _await_or_log(
                    run_preview(
                        task_id, question, redis_stream,
                        vendor=opts.vendor, model=opts.model,
                    ),
                    label=f"preview task={task_id}",
                ),
                name=f"inference-preview:{task_id}",
            ))

        if opts.skills_enabled:
            from .skills_runner import run as run_skills

            bg_tasks.append(asyncio.create_task(
                _await_or_log(
                    run_skills(
                        task_id, question, redis_stream,
                        vendor=opts.vendor, model=opts.model,
                    ),
                    label=f"skills task={task_id}",
                ),
                name=f"inference-skills:{task_id}",
            ))

        # 检索阻塞主线：拿到结果才能开 react_loop
        chunks = await hybrid_search(
            question, policy_id,
            top_n=opts.top_n, top_m=opts.top_m,
        )
        logger.info(
            "[InferencePipeline] task=%s 检索到 %d 个 chunk", task_id, len(chunks)
        )

        await run_react_loop(
            task_id, question, chunks, redis_stream,
            vendor=opts.vendor, model=opts.model,
            intermediate_think_enabled=opts.intermediate_think_enabled,
        )

        # react_loop 结束后等 preview / skills 收尾，确保聚合 think 包含它们最终内容。
        if bg_tasks:
            await asyncio.gather(*bg_tasks, return_exceptions=True)

        await redis_stream.set_status(task_id, "done")
    except asyncio.CancelledError:
        logger.warning("[InferencePipeline] task=%s 被取消", task_id)
        for t in bg_tasks:
            t.cancel()
        await redis_stream.set_status(task_id, "failed", error="cancelled")
        raise
    except Exception as e:
        logger.exception("[InferencePipeline] task=%s 失败: %s", task_id, e)
        for t in bg_tasks:
            t.cancel()
        await redis_stream.set_status(task_id, "failed", error=str(e)[:500])

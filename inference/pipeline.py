"""inference 主调度：preview / skills / hybrid 检索 / ReAct 多轮推理。

设计：

- preview / skills / 检索三件并发跑（``asyncio.create_task``）；
- 检索结果回来后立即开 react_loop；
- preview / skills 即便晚于 react_loop 启动也无所谓，react_loop 在每轮内会重读
  redis 里的最新 preview / skills 快照；
- **SSE 段间锁**：bg_tasks（preview/skills）被透传给 react_loop，让 react_loop 在
  进 final/forced_final 之前 ``await`` 它们收尾，再翻 ``react.intermediateLocked``
  让 SSE 聚合解锁 final 段——这样客户端看到的 think 字段是 append-only，前面已经
  吐过的内容不会被改写。``preview_enabled=False`` / ``skills_enabled=False`` 时本模块
  会立刻把对应的 ``preview.done`` / ``skillsDone`` 翻 True 解锁后续段，避免卡住；
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
    vendor: str = "servyou"
    model: str = "deepseek-v3.2-1163259bcc6c"
    preview_enabled: bool = True
    skills_enabled: bool = True
    top_n: int = 20
    top_m: int = 20
    intermediate_think_enabled: Optional[bool] = None  # None=用全局开关
    # 透传到 preview 的【专题通用知识】槽位（对应 ReasonRequest.answerSystemPrompt）；
    # None / 空字符串时 preview 走兜底占位，不影响原有行为。
    topic_general_knowledge: Optional[str] = None
    # preview 阶段 case 库召回参数；case_top_k=0 表示关闭 case 检索，preview 走
    # 原 2 套 PREVIEW_* / PREVIEW_*_WITH_TGK；>0 时按 case_sim_threshold 过滤
    # cosine_similarity 后取 top-k，走带【历史经验】的新 prompt（详见
    # :func:`inference.prompts.select_preview_prompt`）。
    case_top_k: int = 3
    case_sim_threshold: float = 0.85
    # reSearch 开关：True 时 react_loop 走"动作驱动"状态机（中间轮 incomplete 可选
    # paginate / research），research 分支复用 hybrid_search 重新召回并对已看过证据
    # 做单向去重；False 时完全沿用旧翻页逻辑与旧中间轮 prompt（默认见 app 层入参）。
    re_search_enabled: bool = True
    # 专题定位多候选时的完整 policyId 列表，供 preview 把 case 检索 fan-out 到所有
    # 专题各自 case_{khCode} collection（并发独立召回后合并去重）。None / 单元素时
    # case 检索回落到单集合（用 policy_id），与改造前等价。主推理 pipeline 仍只用
    # 降级取首项的 policy_id，不受此字段影响。
    case_policy_ids: Optional[list[str]] = None


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
                        topic_general_knowledge=opts.topic_general_knowledge,
                        policy_id=policy_id,
                        case_policy_ids=opts.case_policy_ids,
                        case_top_k=opts.case_top_k,
                        case_sim_threshold=opts.case_sim_threshold,
                    ),
                    label=f"preview task={task_id}",
                ),
                name=f"inference-preview:{task_id}",
            ))
        else:
            # preview 不跑时直接翻 done 标志，否则 SSE 聚合的"中间 react 段"门控
            # （需要 preview.done && skillsDone）永远不成立，前端看不到 react 段。
            try:
                await redis_stream.set_preview_done(task_id, True)
            except Exception as e:
                logger.warning(
                    "[InferencePipeline] task=%s set_preview_done(skip) 失败: %s",
                    task_id, e,
                )

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
        else:
            # 同 preview，skills 跳过时立即翻 skillsDone 解锁后续段。
            try:
                await redis_stream.set_skills_done(task_id, True)
            except Exception as e:
                logger.warning(
                    "[InferencePipeline] task=%s set_skills_done(skip) 失败: %s",
                    task_id, e,
                )

        # 检索阻塞主线：拿到结果才能开 react_loop
        chunks = await hybrid_search(
            question, policy_id,
            top_n=opts.top_n, top_m=opts.top_m,
        )
        logger.info(
            "[InferencePipeline] task=%s 检索到 %d 个 chunk", task_id, len(chunks)
        )

        # bg_tasks 传给 react_loop：它会在进 final/forced_final 之前 await，保证
        # final prompt 拿到完整的 preview / skills 快照；同时翻 intermediateLocked
        # 让 SSE 聚合解锁 final 段。中间轮跟 preview/skills 完全并发跑，性能不降。
        await run_react_loop(
            task_id, question, chunks, redis_stream,
            vendor=opts.vendor, model=opts.model,
            intermediate_think_enabled=opts.intermediate_think_enabled,
            bg_tasks=bg_tasks or None,
            # reSearch：research 分支需要 policy_id（带 __cs 后缀的服务端表名）+ top_n/top_m
            # 复用 hybrid_search 重新召回；False 时下面这些参数对 react_loop 无影响。
            re_search_enabled=opts.re_search_enabled,
            policy_id=policy_id,
            top_n=opts.top_n,
            top_m=opts.top_m,
        )

        # react_loop 内部已经在进 final 前 await 过 bg_tasks，这里再 gather 一次
        # 兜底（理论上立即返回）：若 react_loop 走异常路径未能 await，避免泄漏 task。
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

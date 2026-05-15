"""Step2：调度 ``skills.evaluate_and_run`` 并把结果落到 redis 快照。

设计：

- 不修改 ``skills/`` 下任何文件；直接复用 ``evaluate_and_run`` + ``SkillResultRegistry``。
- 完成后调 :meth:`RedisStream.set_skills` 一次性写入；中间过程不流式更新（skills 默认串行执行
  时长可控；后续若要细粒度回显，再换成 per-skill 写入）。
"""

from __future__ import annotations

import logging
from typing import Optional

from skills import (
    SkillResultRegistry,
    SkillRunner,
    evaluate_and_run,
)

from .redis_stream import RedisStream

logger = logging.getLogger(__name__)


def _serialize_registry(registry: SkillResultRegistry) -> list[dict]:
    out: list[dict] = []
    for rec in registry.get_all():
        result = rec.result
        out.append({
            "name": rec.skill_name,
            "command": rec.command,
            "success": bool(result.success),
            "stdout": result.stdout or "",
            "stderr": getattr(result, "stderr", "") or "",
            "exitCode": int(result.exit_code),
        })
    return out


async def run(
    task_id: str,
    question: str,
    redis_stream: RedisStream,
    *,
    vendor: str = "aliyun",
    model: str = "deepseek-v3.2",
    skill_names: Optional[list[str]] = None,
) -> list[dict]:
    """执行 skills 阶段，返回序列化后的 skill 结果列表（已写入 redis）。

    抛出异常时仅记录 + 写空 skills，避免阻断主流水线。

    无论成功或异常路径，``finally`` 都会把 ``skillsDone`` 翻成 True；这是 SSE 聚合
    侧"段间锁"的必要门控：``preview.done && skillsDone`` 才允许 react 中间轮被聚合
    展示，缺一会导致前端永远看不到 react 段。
    """

    registry, runner = SkillResultRegistry(), SkillRunner()
    payload: list[dict] = []
    try:
        try:
            executed = await evaluate_and_run(
                question,
                registry,
                runner,
                vendor=vendor,
                model=model,
                skill_names=skill_names,
            )
            logger.info(
                "[InferenceSkills] task=%s 已执行 skills=%s 共 %d 条记录",
                task_id, executed, len(registry.get_all()),
            )
        except Exception as e:
            logger.exception("[InferenceSkills] task=%s evaluate_and_run 失败: %s", task_id, e)

        payload = _serialize_registry(registry)
        await redis_stream.set_skills(task_id, payload)
        return payload
    finally:
        # 兜底：哪怕 evaluate_and_run / set_skills 异常退出，也要解锁 SSE 聚合侧的
        # 后续段，避免前端永远卡在 preview 段。
        try:
            await redis_stream.set_skills_done(task_id, True)
        except Exception as e:
            logger.warning(
                "[InferenceSkills] task=%s set_skills_done 失败（忽略）: %s",
                task_id, e,
            )

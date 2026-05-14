"""【专题Know How定位】外部 SSE 客户端封装。

仅供 :mod:`app` 在 ``/api/inference/stream`` 路由用户**未指定 policyId** 时调用，
作用是先把问题丢到外部分类服务拿到匹配的"业务专题（ywzt）+ policyId"，再用解析
出的 policyId 走原有 inference pipeline。

外部接口：

- URL: ``http://mlp.paas.dc.servyou-it.com/kh_classify/ywzt_classify_stream`` (POST)
- 入参 body: ``{"question": "<用户原始问题>"}``
- 响应: SSE 流，每行一个 JSON（部分网关带 ``data: `` 前缀，部分裸 JSON），形如::

    {"status": "running", "data": {"id": "single", "question": "...",
        "answer": "", "ywzt": [], "policyId": [], "reasoning": "..."}}

  ``status`` 取值 ``"start"`` / ``"running"`` / ``"finish"``；
  ``data.reasoning`` 是模型的全量推理累积串（**每帧覆盖**，非增量 delta）；
  终态 ``status == "finish"`` 时 ``data.ywzt`` 与 ``data.policyId`` 才被填充
  （均为 list；当前阶段只接受唯一命中）。

本模块只做两件事：

1. 流式过程中把每帧 ``data.reasoning`` 通过 ``RedisStream.append_topic_locate_reasoning``
   覆盖式写到 redis 快照，由 ``recompute_aggregates`` 把它前置拼到接口 ``think``;
2. ``status=finish`` 时按 ``len(ywzt)==1 and len(policyId)==1`` 判定:
   - 唯一命中：调 ``set_topic_locate_done`` 触发 ``###【专题Know How定位】\\n\\t{ywzt}``
     收口段渲染，并把 ``(policyId, ywzt, None)`` 返回给调用方继续走 inference;
   - 0 个或多个候选：返回 ``(None, None, refusal_text)``，由调用方写
     ``topicLocate.refusal`` + ``status=done`` 直接拒答。

任何阶段的网络/解析异常都向上抛，由调用方走拒答兜底。
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import httpx

from .redis_stream import RedisStream

logger = logging.getLogger(__name__)


# 接口固定不变；如果后续要做环境覆盖再考虑放到 inference.config。
_TOPIC_LOCATOR_URL = (
    "http://mlp.paas.dc.servyou-it.com/kh_classify/ywzt_classify_stream"
)

_DEFAULT_TIMEOUT_SECONDS: float = 120.0
_DEFAULT_CONNECT_TIMEOUT_SECONDS: float = 15.0


# ---------------------------------------------------------------- SSE 行解析

def _parse_sse_line(line: str) -> Optional[dict]:
    """解析单行外部 SSE。返回 ``None`` 表示忽略（注释/空行/解析失败）。

    兼容两种网关行为：
    - 标准 SSE：``data: {...json...}``（``data:`` 前缀，可有空格）；
    - 裸 JSON 行：直接 ``{...json...}``（用户实测样例）。

    任何 ``[DONE]`` 之类的终结标记也归为 ``None``。
    """

    if not line:
        return None
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if line.startswith("data:"):
        payload = line[len("data:"):].strip()
    else:
        payload = line
    if not payload or payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("[TopicLocator] SSE 行解析失败: %r", payload[:200])
        return None
    if not isinstance(obj, dict):
        return None
    return obj


# ---------------------------------------------------------------- 拒答文案
#
# TODO: 暂行策略——候选业务专题非唯一时直接拒答。
# 后续可能改为：多候选时并行跑多套 inference 取首个/取最高分；0 候选时回落到
# 一个"通用 policyId"等。改动时同步修改本函数与 ``app._run_inference_with_topic_locator``
# 里对 refusal 的处理。

def _build_refusal(ywzt: list, policy_id: list) -> str:
    """根据 ywzt/policyId 的命中数量生成拒答文案。"""

    if not ywzt or not policy_id:
        return (
            "未能定位到匹配的业务专题，无法继续推理，"
            "请补充更具体的问题描述或显式指定 policyId。"
        )
    return (
        f"问题命中多个业务专题（{', '.join(str(x) for x in ywzt)}），"
        "暂不支持多专题并行推理，请进一步明确意图或显式指定 policyId。"
    )


# ---------------------------------------------------------------- 主入口


async def run_topic_locator(
    task_id: str,
    question: str,
    redis_stream: RedisStream,
    *,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    connect_timeout_seconds: float = _DEFAULT_CONNECT_TIMEOUT_SECONDS,
    client: Optional[httpx.AsyncClient] = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """流式调用【专题Know How定位】并把 reasoning 转发到 redis 快照。

    返回 ``(policy_id, ywzt, refusal_message)``：

    - 唯一命中：``(policyId[0], ywzt[0], None)``，并已经把 done=True/ywzt 写到快照,
      调用方应继续用该 ``policyId`` 跑 inference pipeline；
    - 0/多个候选：``(None, None, refusal_text)``，调用方应写 ``topicLocate.refusal``
      并把任务置为 ``status=done`` 直接拒答；
    - 网络/HTTP/解析异常：直接向上抛 ``Exception`` 由调用方兜底。

    与 :func:`inference.llm_stream.chat_stream` 风格一致：``client`` 可外部注入复用
    连接池，不传则本地 new 一个并自动 close。
    """

    timeout = httpx.Timeout(
        timeout=timeout_seconds,
        connect=connect_timeout_seconds,
        read=timeout_seconds,
        write=connect_timeout_seconds,
        pool=connect_timeout_seconds,
    )
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)

    # 仅当 finish 帧出现且唯一命中时填充。
    final_ywzt: Optional[str] = None
    final_policy_id: Optional[str] = None
    refusal: Optional[str] = None
    saw_finish = False

    try:
        async with cli.stream(
            "POST",
            _TOPIC_LOCATOR_URL,
            json={"question": question},
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", errors="replace")
                raise httpx.HTTPStatusError(
                    f"topic_locator http={resp.status_code} body={body[:500]}",
                    request=resp.request,
                    response=resp,
                )
            async for raw_line in resp.aiter_lines():
                obj = _parse_sse_line(raw_line)
                if obj is None:
                    continue
                status = (obj.get("status") or "").strip().lower()
                data = obj.get("data") or {}
                # reasoning 每帧覆盖（外部接口推送的就是当前累积全量）。
                reasoning = data.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    try:
                        await redis_stream.append_topic_locate_reasoning(
                            task_id, reasoning
                        )
                    except Exception as e:
                        # 写一帧失败不致命，下一帧还会再覆盖一次全量。
                        logger.warning(
                            "[TopicLocator] task=%s 写 reasoning 失败（忽略）: %s",
                            task_id, e,
                        )

                if status == "finish":
                    saw_finish = True
                    ywzt_list = data.get("ywzt") or []
                    pid_list = data.get("policyId") or []
                    if not isinstance(ywzt_list, list):
                        ywzt_list = []
                    if not isinstance(pid_list, list):
                        pid_list = []
                    if len(ywzt_list) == 1 and len(pid_list) == 1:
                        final_ywzt = (str(ywzt_list[0]) or "").strip()
                        final_policy_id = (str(pid_list[0]) or "").strip()
                        if not final_ywzt or not final_policy_id:
                            # 命中但取值空：当作未命中拒答。
                            refusal = _build_refusal(ywzt_list, pid_list)
                            final_ywzt = None
                            final_policy_id = None
                    else:
                        refusal = _build_refusal(ywzt_list, pid_list)
                    # finish 帧已经携带终态，可以提前 break；服务端也会在下一刻关流。
                    break
    finally:
        if own_client:
            await cli.aclose()

    if not saw_finish:
        # 流被中途切断也按拒答兜底（与 0 候选同语义）。
        return None, None, _build_refusal([], [])

    if final_policy_id and final_ywzt:
        try:
            await redis_stream.set_topic_locate_done(task_id, final_ywzt)
        except Exception as e:
            logger.warning(
                "[TopicLocator] task=%s 写 topicLocate.done 失败（忽略）: %s",
                task_id, e,
            )
        return final_policy_id, final_ywzt, None

    return None, None, refusal or _build_refusal([], [])

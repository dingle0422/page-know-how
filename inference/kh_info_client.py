"""【专题Know How信息】查询客户端封装。

仅供 :mod:`app` 在 ``/api/inference/stream`` 路由 **确认 policyId 之后** 调用，
作用是按 policyId 拉取对应专题的「通用知识」prompt 文本（``data.prompt`` 字段），
注入到 preview 阶段的 ``topic_general_knowledge`` 槽位（见
:func:`inference.prompts.select_preview_prompt`）。

外部接口：

- URL: ``http://10.199.0.40:8080/api/kh/getKhInfoByPolicyId`` (POST)
- 入参 body: ``{"policyId": "<policyId，格式：code_version>"}``
- 响应：标准 JSON，``data.prompt`` 即为本模块所需的通用知识文本（其余字段当前不使用）。

异常策略：网络/HTTP/解析/字段缺失任何环节出问题，统一返回 ``None`` 并打 warning，
**不向上抛**——调用方可据此决定是否回落到 ``ReasonRequest.answerSystemPrompt``
（即"未自动获取到时仍允许沿用客户端显式传入的专题通用知识"）。
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# 接口固定不变；如果后续要做环境覆盖再考虑放到 inference.config。
_KH_INFO_URL = "http://10.199.0.40:8080/api/kh/getKhInfoByPolicyId"

# 这是一个相对轻量的查询接口，留出 10s 已经很宽松；超时即视作失败回落。
_DEFAULT_TIMEOUT_SECONDS: float = 10.0
_DEFAULT_CONNECT_TIMEOUT_SECONDS: float = 5.0


async def fetch_topic_general_knowledge(
    policy_id: str,
    *,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    connect_timeout_seconds: float = _DEFAULT_CONNECT_TIMEOUT_SECONDS,
    client: Optional[httpx.AsyncClient] = None,
) -> Optional[str]:
    """按 ``policyId`` 拉取专题通用知识（``data.prompt``）。

    返回非空字符串表示拿到了；返回 ``None`` 表示接口失败 / 字段缺失 / 字段为空字符串，
    调用方应回落到 ``ReasonRequest.answerSystemPrompt``（仍可能为 ``None``，
    此时 preview 走原版 ``PREVIEW_*`` prompt，与改造前完全等价）。

    与 :func:`inference.topic_locator.run_topic_locator` 风格一致：``client`` 可外部
    注入复用连接池，不传则本地 new 一个并自动 close。
    """

    pid = (policy_id or "").strip()
    if not pid:
        return None

    timeout = httpx.Timeout(
        timeout=timeout_seconds,
        connect=connect_timeout_seconds,
        read=timeout_seconds,
        write=connect_timeout_seconds,
        pool=connect_timeout_seconds,
    )
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)

    try:
        resp = await cli.post(
            _KH_INFO_URL,
            json={"policyId": pid},
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code >= 400:
            logger.warning(
                "[KhInfo] policyId=%s HTTP %s body=%r",
                pid, resp.status_code, resp.text[:500],
            )
            return None
        try:
            obj = resp.json()
        except Exception as e:
            logger.warning(
                "[KhInfo] policyId=%s JSON 解析失败: %s body=%r",
                pid, e, resp.text[:500],
            )
            return None
        if not isinstance(obj, dict):
            logger.warning("[KhInfo] policyId=%s 返回体非 dict: %r", pid, obj)
            return None
        # 接口规范里 success/code 都有，但只要 data.prompt 非空就算拿到——
        # 兼容上游偶发的 success=true 但 data 为 null / data.prompt 为空的边界场景。
        data = obj.get("data")
        if not isinstance(data, dict):
            logger.info(
                "[KhInfo] policyId=%s data 字段缺失或非 dict，success=%r message=%r",
                pid, obj.get("success"), obj.get("message"),
            )
            return None
        prompt = data.get("prompt")
        if not isinstance(prompt, str):
            return None
        prompt = prompt.strip()
        return prompt or None
    except Exception as e:
        logger.warning("[KhInfo] policyId=%s 调用异常: %s", pid, e)
        return None
    finally:
        if own_client:
            await cli.aclose()

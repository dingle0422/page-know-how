"""LLM 流式 SSE 客户端。

与 ``llm/client.py`` 的 ``chat()`` 共用 vendor 路由约定，但实现走 ``httpx`` 异步流式：

- ``stream=True`` 的 OpenAI 兼容 SSE：每行 ``data: {...}``，结束行 ``data: [DONE]``。
- 单条 chunk 内 ``choices[0].delta.content`` -> 产 ``("answer", str)``；
- ``choices[0].delta.reasoning_content`` -> 产 ``("think", str)``。

不导入 ``llm/client.py``、不修改老接口；上层 ReAct/preview 阶段统一用本模块。

主要导出：

- :func:`chat_stream`：异步生成器，吐 ``(channel, delta)``。
- :class:`StreamTagRouter`：纯文本解析器，按 ``<think>...</think>`` / ``<answer>...</answer>`` /
  ``<verdict>...</verdict>`` 边界切分增量，配合 vendor 把 think/answer 全塞 content
  的情况下使用。
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)

# 与 llm/client.py 的网关授权信息保持一致；后续若改成 env 注入，两处一起改即可。
_DEFAULT_APP_ID = "sk-0609aa6d08de4413a72e14b3fb8fbab1"
_DEEPSEEK_APP_ID = "sk-0609aa6d08de4413a72e14b3fb8fbab1"
_SERVYOU_APP_ID = "sk-d75b519b704d4d348245efe435f08ff3"


# ---------------------------------------------------------------- vendor 路由

def _build_request(
    messages_payload: list[dict],
    *,
    vendor: str,
    model: str,
    enable_thinking: bool,
) -> tuple[str, dict, dict]:
    """根据 vendor 返回 ``(url, headers, payload)``，统一把 ``stream=True`` 打开。

    与 ``llm/client.py:chat`` 的分支保持一致；任何新增 vendor 都应在两处同步。
    """

    if vendor == "qwen3.5-122b-a10b":
        url = "http://211.137.21.19:17860/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "Qwen3.5-122B-A10B",
            "messages": messages_payload,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
    elif vendor == "qwen3.6-35b-a3b":
        url = "http://mlp.paas.dc.servyou-it.com/qwen3.6-35b-a3b/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "Qwen/Qwen3.6-35B-A3B",
            "messages": messages_payload,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
    elif vendor == "qwen3.5-27b":
        url = "http://mlp.paas.dc.servyou-it.com/qwen3.5-27b/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "Qwen/Qwen3.5-27B",
            "messages": messages_payload,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
    elif vendor in ("deepseek-v4-flash", "deepseek-v4-pro"):
        url = "http://mlp.paas.dc.servyou-it.com/mudgate/api/llm/deepseek/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": _DEEPSEEK_APP_ID}
        payload = {
            "appId": _DEEPSEEK_APP_ID,
            "model": vendor,
            "messages": messages_payload,
            "stream": True,
        }
        if enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
    else:
        if vendor == "servyou":
            url = f"http://10.199.0.7:5000/api/llm/{vendor}/v1/chat/completions"
            app_id = _SERVYOU_APP_ID
        else:
            url = f"http://mlp.paas.dc.servyou-it.com/mudgate/api/llm/{vendor}/v1/chat/completions"
            app_id = _DEFAULT_APP_ID
        headers = {"Content-Type": "application/json", "Authorization": app_id}
        payload = {
            "appId": app_id,
            "model": model,
            "messages": messages_payload,
            "stream": True,
        }
        if enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}

    return url, headers, payload


# ---------------------------------------------------------------- SSE 解析

def _parse_sse_line(line: str) -> Optional[dict]:
    """解析单行 OpenAI 兼容 SSE。

    返回 ``None`` 表示忽略（注释、空行、结束标记或解析失败）。
    """

    if not line:
        return None
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("LLM SSE 行解析失败: %r", payload[:200])
        return None


def _extract_deltas(chunk: dict) -> list[tuple[str, str]]:
    """从单条 SSE chunk 中抽出 (channel, delta) 列表。

    OpenAI 兼容协议下 ``chunk["choices"][i]["delta"]`` 可能同时含
    ``reasoning_content``（think）与 ``content``（answer），按出现顺序产出。
    """

    out: list[tuple[str, str]] = []
    choices = chunk.get("choices") or []
    for ch in choices:
        delta = (ch or {}).get("delta") or {}
        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            out.append(("think", reasoning))
        content = delta.get("content")
        if isinstance(content, str) and content:
            out.append(("answer", content))
    return out


# ---------------------------------------------------------------- 主入口


async def chat_stream(
    messages: str,
    *,
    vendor: str = "qwen3.5-122b-a10b",
    model: str = "Qwen3.5-122B-A10B",
    system: Optional[str] = None,
    enable_thinking: bool = False,
    timeout_seconds: float = 360.0,
    connect_timeout_seconds: float = 30.0,
    client: Optional[httpx.AsyncClient] = None,
) -> AsyncIterator[tuple[str, str]]:
    """异步流式调用 LLM，按 vendor 路由生成 ``(channel, delta)``。

    - ``channel`` 为 ``"think"`` 或 ``"answer"``。
    - 自动跳过 SSE 注释与 ``[DONE]``。
    - 网关返回 4xx/5xx 时会主动 raise ``httpx.HTTPStatusError``。
    - ``client`` 可外部注入复用连接池；不传则本地 new 一个，结束自动 close。

    注意：vendor 把所有内容都塞 content（不吐 reasoning_content）的情况下，
    ``<think>...</think>`` 会作为 content 流式过来，需要上层用 :class:`StreamTagRouter`
    做二次 dispatch；本函数本身只忠实分发底层协议字段。
    """

    messages_payload: list[dict] = []
    if system:
        messages_payload.append({"role": "system", "content": system})
    messages_payload.append({"role": "user", "content": messages})

    url, headers, payload = _build_request(
        messages_payload,
        vendor=vendor,
        model=model,
        enable_thinking=enable_thinking,
    )

    timeout = httpx.Timeout(
        timeout=timeout_seconds,
        connect=connect_timeout_seconds,
        # 流式连接整体生命周期可能较长，把 read 也放宽到整体 timeout
        read=timeout_seconds,
        write=connect_timeout_seconds,
        pool=connect_timeout_seconds,
    )

    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout, headers=headers)
    try:
        async with cli.stream(
            "POST",
            url,
            headers=headers,
            json=payload,
            timeout=timeout,
        ) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", errors="replace")
                raise httpx.HTTPStatusError(
                    f"LLM stream {vendor}/{model} http={resp.status_code} body={body[:500]}",
                    request=resp.request,
                    response=resp,
                )
            async for raw_line in resp.aiter_lines():
                chunk = _parse_sse_line(raw_line)
                if chunk is None:
                    continue
                # 兼容部分老网关把错误塞在 chunk["success"]=false 字段里
                if isinstance(chunk, dict) and "success" in chunk and not chunk.get("success"):
                    err = chunk.get("errorContext") or chunk.get("message") or "unknown"
                    raise RuntimeError(f"LLM stream {vendor}/{model} 网关错误: {err}")
                for item in _extract_deltas(chunk):
                    yield item
    finally:
        if own_client:
            await cli.aclose()


# ---------------------------------------------------------------- 标签状态机


class StreamTagRouter:
    """把含 ``<think>...</think>`` / ``<answer>...</answer>`` / ``<verdict>...</verdict>``
    的 *answer 通道* 增量按标签拆分到 think/answer/verdict 三个子流。

    使用场景：vendor 不吐 ``reasoning_content`` 而是把 think 标签直接塞 content 时，
    用本类把 :func:`chat_stream` 产生的 ``("answer", delta)`` 再按标签 dispatch。

    设计：

    - 完全增量（``feed(text)`` 可以拿到一个字符也照常工作），不依赖完整 chunk 边界。
    - 不解析嵌套：``<think>`` 内部不允许再出现 ``<think>``。
    - 遇到 ``<verdict>`` 内容只缓存到 ``self.verdict``，不分发到回调；外层在轮末读取。

    ``feed()`` 会调用注入的 ``on_think(delta)`` / ``on_answer(delta)`` 回调；
    任一回调可以是同步函数也可以是 awaitable（本类不去 await，由调用方决定异步包装）。
    """

    _TAGS = {
        "<think>": ("think", True),
        "</think>": ("think", False),
        "<answer>": ("answer", True),
        "</answer>": ("answer", False),
        "<verdict>": ("verdict", True),
        "</verdict>": ("verdict", False),
    }

    def __init__(self) -> None:
        self._buffer = ""
        self._mode: Optional[str] = None  # think / answer / verdict / None
        self.verdict: str = ""

    def _max_pending_partial_tag_len(self) -> int:
        # 最长标签为 "</verdict>"，长度 10。预留缓存避免标签被切两半时漏判。
        return 10

    def feed(
        self,
        text: str,
        *,
        on_think: Optional[callable] = None,
        on_answer: Optional[callable] = None,
    ) -> None:
        """喂入新增量并触发回调。"""

        if not text:
            return
        self._buffer += text

        # 循环：每次找最近的一个完整标签，处理标签前的纯文本，然后切换状态。
        while True:
            # 找最近的 "<"。如果没有 "<"，整段都是文本，按当前 mode 派发掉，留底兜底。
            lt = self._buffer.find("<")
            if lt < 0:
                self._dispatch(self._buffer, on_think=on_think, on_answer=on_answer)
                self._buffer = ""
                return

            # 派发 "<" 之前的纯文本。
            if lt > 0:
                self._dispatch(self._buffer[:lt], on_think=on_think, on_answer=on_answer)
                self._buffer = self._buffer[lt:]
                lt = 0

            # 在 "<" 之后找 ">"；没有就保留 buffer 等下一次 feed。
            gt = self._buffer.find(">")
            if gt < 0:
                # 半截标签：避免 buffer 一直涨；超过最大标签长仍无 ">"，
                # 视为普通字符按当前 mode 吐出去。
                if len(self._buffer) > self._max_pending_partial_tag_len():
                    head = self._buffer[:1]
                    self._dispatch(head, on_think=on_think, on_answer=on_answer)
                    self._buffer = self._buffer[1:]
                    continue
                return

            tag = self._buffer[: gt + 1].lower()
            if tag in self._TAGS:
                kind, opening = self._TAGS[tag]
                if opening:
                    self._mode = kind
                else:
                    if self._mode == kind:
                        self._mode = None
                self._buffer = self._buffer[gt + 1:]
                continue

            # 不是已知标签，把 "<" 当成普通字符派发掉，避免死循环。
            self._dispatch(self._buffer[:1], on_think=on_think, on_answer=on_answer)
            self._buffer = self._buffer[1:]

    def _dispatch(
        self,
        text: str,
        *,
        on_think: Optional[callable],
        on_answer: Optional[callable],
    ) -> None:
        if not text:
            return
        if self._mode == "think":
            if on_think is not None:
                on_think(text)
        elif self._mode == "answer":
            if on_answer is not None:
                on_answer(text)
        elif self._mode == "verdict":
            self.verdict += text
        # mode is None：标签外的散文本直接丢弃（一般是模型在 <answer> 之外的口胡）。

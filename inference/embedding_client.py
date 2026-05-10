"""外部 embedding 服务封装（BGE）。

接口契约（已实测）::

    POST http://mlp.paas.dc.servyou-it.com/text-embedding-bge/embedding
    Headers: {"Content-Type": "application/json; charset=UTF-8"}
    Body:    {"sentences": ["...", "..."]}
    200 OK   {"code": 200, "msg": "success",
              "output": [{"embedding": [1024 floats], "text_index": 0}, ...]}

设计要点：

- URL / model 由环境变量覆盖，但默认就是上述生产地址，**开箱即用**，不再需要先配 env。
- 响应里 ``output[i]`` 用 ``text_index`` 标注与输入的对应关系；本封装会按 ``text_index``
  显式排序回原顺序，避免后端任何乱序情况。
- 显式判 ``code == 200``，失败抛 ``RuntimeError`` 并附 msg；HTTP 4xx/5xx 直接 raise。
- 提供 ``INFERENCE_EMBEDDING_DISABLED=1`` 兜底开关：上层（hybrid 检索 / indexer）
  会捕获 :class:`EmbeddingNotConfigured` 优雅降级到纯 BM25。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


_DEFAULT_URL = "http://mlp.paas.dc.servyou-it.com/text-embedding-bge/embedding"
_DEFAULT_MODEL = "bge"


class EmbeddingNotConfigured(RuntimeError):
    """显式禁用 embedding（``INFERENCE_EMBEDDING_DISABLED=1``）时抛出。

    上层（``inference.retrieval.semantic``、``inference.retrieval.indexer``）应
    捕获本异常并降级。
    """


def _read_env() -> tuple[str, str, Optional[str]]:
    if os.getenv("INFERENCE_EMBEDDING_DISABLED", "0").lower() in {"1", "true", "yes", "on"}:
        raise EmbeddingNotConfigured(
            "INFERENCE_EMBEDDING_DISABLED=1，本进程显式禁用 embedding"
        )
    url = os.getenv("INFERENCE_EMBEDDING_URL", _DEFAULT_URL).strip() or _DEFAULT_URL
    model = os.getenv("INFERENCE_EMBEDDING_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    auth = os.getenv("INFERENCE_EMBEDDING_AUTH", "").strip() or None
    return url, model, auth


def _parse_response(body: dict, batch_len: int) -> list[list[float]]:
    """把单次响应解析成与请求顺序对齐的 ``list[list[float]]``。"""

    code = body.get("code")
    if code is not None and int(code) != 200:
        raise RuntimeError(
            f"embedding 服务返回失败 code={code} msg={body.get('msg')!r}"
        )
    output = body.get("output")
    if not isinstance(output, list):
        raise RuntimeError(
            f"embedding 响应缺少 output 列表，实际 keys={list(body.keys())}"
        )
    if len(output) != batch_len:
        raise RuntimeError(
            f"embedding 响应数量不匹配 expected={batch_len} got={len(output)}"
        )
    # 用 text_index 显式回到原顺序。若后端不返回 text_index，则按列表顺序兜底。
    placeholders: list[Optional[list[float]]] = [None] * batch_len
    for pos, item in enumerate(output):
        if not isinstance(item, dict):
            raise RuntimeError(f"output[{pos}] 不是 dict: {type(item).__name__}")
        emb = item.get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError(f"output[{pos}].embedding 类型异常: {type(emb).__name__}")
        idx_raw = item.get("text_index")
        idx = int(idx_raw) if idx_raw is not None else pos
        if not (0 <= idx < batch_len):
            raise RuntimeError(f"output[{pos}].text_index={idx} 越界 batch_len={batch_len}")
        if placeholders[idx] is not None:
            raise RuntimeError(f"output 中 text_index={idx} 重复出现")
        placeholders[idx] = [float(x) for x in emb]
    if any(v is None for v in placeholders):
        missing = [i for i, v in enumerate(placeholders) if v is None]
        raise RuntimeError(f"embedding 响应缺失 text_index={missing}")
    return placeholders  # type: ignore[return-value]


async def embed_texts(
    texts: list[str],
    *,
    model: Optional[str] = None,  # 兼容签名，目前后端按 URL 路径区分模型，本字段仅记录
    batch_size: int = 32,
    timeout_seconds: float = 60.0,
    client: Optional[httpx.AsyncClient] = None,
) -> list[list[float]]:
    """异步批量获取 embedding，输出顺序与 ``texts`` 保持一致。

    - 自动按 ``batch_size`` 分批；
    - 任何 HTTP 异常 / 业务 code 非 200 都直接 raise，调用方决定是否降级。
    """

    if not texts:
        return []
    url, default_model, auth = _read_env()
    use_model = model or default_model
    headers = {"Content-Type": "application/json; charset=UTF-8"}
    if auth:
        headers["Authorization"] = auth

    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout_seconds, headers=headers)

    out: list[list[float]] = []
    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            payload = {"sentences": batch}
            resp = await cli.post(url, json=payload, headers=headers, timeout=timeout_seconds)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"embedding HTTP {resp.status_code} body={resp.text[:500]}"
                )
            body = resp.json()
            out.extend(_parse_response(body, len(batch)))
    finally:
        if own_client:
            await cli.aclose()

    logger.debug(
        "[Embedding] 取得 %d 条向量 (model=%s, dim=%d)",
        len(out), use_model, len(out[0]) if out else 0,
    )
    return out


def embed_texts_sync(
    texts: list[str],
    *,
    model: Optional[str] = None,
    batch_size: int = 32,
    timeout_seconds: float = 60.0,
) -> list[list[float]]:
    """同步版本，便于离线脚本（如建索引 CLI）直接调用而不必绕 asyncio。"""

    if not texts:
        return []
    import requests  # 延迟导入：异步路径不需要

    url, default_model, auth = _read_env()
    use_model = model or default_model
    headers = {"Content-Type": "application/json; charset=UTF-8"}
    if auth:
        headers["Authorization"] = auth

    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        payload = {"sentences": batch}
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout_seconds)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"embedding HTTP {resp.status_code} body={resp.text[:500]}"
            )
        body = resp.json()
        out.extend(_parse_response(body, len(batch)))

    logger.debug(
        "[Embedding] (sync) 取得 %d 条向量 (model=%s, dim=%d)",
        len(out), use_model, len(out[0]) if out else 0,
    )
    return out

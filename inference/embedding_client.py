"""外部 embedding 服务封装（默认 Qwen3-Embedding，兼容旧 BGE 协议）。

默认接口契约（OpenAI Embeddings 风格）::

    POST http://mlp.paas.dc.servyou-it.com/qwen3-embedding/v1/embeddings
    Headers: {"Content-Type": "application/json; charset=UTF-8"}
    Body:    {"input": ["...", "..."], "model": "qwen3-embedding"}
    200 OK   {"object": "list",
              "data": [{"index": 0, "embedding": [N floats], "object": "embedding"}, ...],
              "model": "...", "usage": {...}, "created": ...}

兼容旧 BGE 接口（URL 不含 ``/v1/embeddings`` 时自动识别）::

    POST http://mlp.paas.dc.servyou-it.com/text-embedding-bge/embedding
    Body:    {"sentences": ["...", "..."]}
    200 OK   {"code": 200, "msg": "success",
              "output": [{"embedding": [N floats], "text_index": 0}, ...]}

设计要点：

- URL / model 由环境变量覆盖，但默认就是上述生产地址，**开箱即用**，不再需要先配 env。
- 依据 URL 自动识别协议：``/v1/embeddings`` 走 OpenAI 风格，否则走 BGE 风格。
- 响应里 ``data[i].index``（或 ``output[i].text_index``）用于显式回到原顺序。
- HTTP 4xx/5xx 都会抛错；200 但业务失败（BGE code!=200）也抛错。
- 提供 ``INFERENCE_EMBEDDING_DISABLED=1`` 兜底开关：上层（hybrid 检索 / indexer）
  会捕获 :class:`EmbeddingNotConfigured` 优雅降级到纯 BM25。
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


_DEFAULT_URL = "http://mlp.paas.dc.servyou-it.com/qwen3-embedding/v1/embeddings"
_DEFAULT_MODEL = "qwen3-embedding"

# 服务端硬限制：单次请求 input/sentences 列表长度 ≤ 10，超过会返回
# code=400 msg='List length exceeds the maximum limit of 10.'。
# 客户端无论传多大的 batch_size 都按这里向下夹断，避免直接打到上游报错。
_MAX_BATCH_SIZE = 10

# qwen3-embedding 当前上限：单条 input 文本长度 ≤ 10240（按服务约定）。
# 这里默认对齐到 10240，避免过早截断影响召回；如需更保守可通过环境变量覆盖：
#   INFERENCE_EMBEDDING_MAX_INPUT_CHARS=...
# 截断只发生在送给上游 API 的副本上，调用方原文本（写进 _chunks.jsonl / BM25）不受影响。
_MAX_INPUT_CHARS = max(1, int(os.getenv("INFERENCE_EMBEDDING_MAX_INPUT_CHARS", "10240")))


def _clamp_batch_size(batch_size: int) -> int:
    """把请求方传入的 batch_size 夹到 [1, _MAX_BATCH_SIZE] 范围内。"""

    try:
        bs = int(batch_size)
    except (TypeError, ValueError):
        bs = _MAX_BATCH_SIZE
    if bs <= 0:
        bs = _MAX_BATCH_SIZE
    return min(bs, _MAX_BATCH_SIZE)


def _clip_for_embedding(text: str) -> str:
    """对单条 sentence 做"embedding 安全截断"。

    embedding 模型对单条输入长度有硬上限（见 _MAX_INPUT_CHARS 注释）。
    截断只发生在送给上游 API 的副本上，调用方拿到的 vectors 与原文本一一对应，
    chunk 头部包含【命中关键词】/【关联条款位置】等关键坐标信息，截断后头部
    语义保留完整，召回效果可接受。
    """
    if len(text) <= _MAX_INPUT_CHARS:
        return text
    return text[:_MAX_INPUT_CHARS]


# --- 重试机制 -------------------------------------------------------------
#
# embedding 服务对超大并发或边界 sentence 偶发返回 JSON null / 5xx / 网络瞬断；
# 这类错误大多通过短暂退避后重试即可恢复。本模块按"单批"粒度做重试，避免一次
# 全量重发污染已经成功的批次。
#
# 重试与非重试的边界：
#   可重试：网络层异常（httpx.RequestError / requests.ConnectionError 等）、
#           HTTP 5xx、body=null、业务 code 非 200 的临时错误、JSON 解析失败；
#   不可重试：HTTP 4xx（含上游已知硬性参数错误，例如 'Range of input length'
#             / 'List length exceeds'），重试只是浪费时间。
#
# 通过环境变量覆盖：
#   INFERENCE_EMBEDDING_RETRY_ATTEMPTS：每批最大尝试次数（含首次），默认 3
#   INFERENCE_EMBEDDING_RETRY_BASE_DELAY：首次重试前的基础等待秒数，默认 0.5
#   INFERENCE_EMBEDDING_RETRY_MAX_DELAY：单次重试退避秒数上限，默认 8.0

_RETRY_ATTEMPTS = max(1, int(os.getenv("INFERENCE_EMBEDDING_RETRY_ATTEMPTS", "3")))
_RETRY_BASE_DELAY = max(0.0, float(os.getenv("INFERENCE_EMBEDDING_RETRY_BASE_DELAY", "0.5")))
_RETRY_MAX_DELAY = max(_RETRY_BASE_DELAY, float(os.getenv("INFERENCE_EMBEDDING_RETRY_MAX_DELAY", "8.0")))


class _NonRetryableEmbeddingError(RuntimeError):
    """显式标记的"硬错误"，重试包装层看到这个异常直接 raise 不再重试。

    典型场景：HTTP 4xx（含上游业务硬性参数错误，如 sentence 超长 / batch
    超长），重试无意义。
    """


# 上游业务错误中已知不可重试的关键字（即便 HTTP=200 但 body.code 是它们）。
_NON_RETRYABLE_BUSINESS_PATTERNS = (
    "Range of input length",
    "List length exceeds",
)


def _is_business_error_retryable(msg: str) -> bool:
    """判断业务错误消息是否值得重试。命中硬性参数错误关键字即不重试。"""
    if not msg:
        return True
    return not any(pat in msg for pat in _NON_RETRYABLE_BUSINESS_PATTERNS)


def _retry_delay(attempt: int) -> float:
    """计算第 ``attempt`` 次重试前的等待秒数（attempt 从 1 开始）。

    指数退避（基础 * 2^(n-1)）+ [0, 0.5) 的均匀随机抖动；夹断到 _RETRY_MAX_DELAY。
    随机抖动避免多个并发请求在同一时刻重发挤爆上游。
    """
    base = _RETRY_BASE_DELAY * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0.0, 0.5)
    return min(_RETRY_MAX_DELAY, base + jitter)


def _detect_protocol(url: str) -> str:
    """识别上游协议：openai / bge。

    优先使用 ``INFERENCE_EMBEDDING_PROTOCOL`` 显式覆盖；否则按 URL 自动识别：
    - 含 ``/v1/embeddings`` -> openai
    - 其他 -> bge
    """
    override = os.getenv("INFERENCE_EMBEDDING_PROTOCOL", "").strip().lower()
    if override in {"openai", "bge"}:
        return override
    if override:
        logger.warning(
            "[Embedding] INFERENCE_EMBEDDING_PROTOCOL=%s 非法，回退自动识别", override
        )
    return "openai" if "/v1/embeddings" in url.lower() else "bge"


def _build_payload(batch: list[str], model: str, protocol: str) -> dict:
    if protocol == "openai":
        return {"input": batch, "model": model}
    return {"sentences": batch}


class EmbeddingNotConfigured(RuntimeError):
    """显式禁用 embedding（``INFERENCE_EMBEDDING_DISABLED=1``）时抛出。

    上层（``inference.retrieval.semantic``、``inference.retrieval.indexer``）应
    捕获本异常并降级。
    """


def _read_env() -> tuple[str, str, Optional[str], str]:
    if os.getenv("INFERENCE_EMBEDDING_DISABLED", "0").lower() in {"1", "true", "yes", "on"}:
        raise EmbeddingNotConfigured(
            "INFERENCE_EMBEDDING_DISABLED=1，本进程显式禁用 embedding"
        )
    url = os.getenv("INFERENCE_EMBEDDING_URL", _DEFAULT_URL).strip() or _DEFAULT_URL
    model = os.getenv("INFERENCE_EMBEDDING_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    auth = os.getenv("INFERENCE_EMBEDDING_AUTH", "").strip() or None
    protocol = _detect_protocol(url)
    return url, model, auth, protocol


def _parse_response(body, batch_len: int, protocol: str) -> list[list[float]]:
    """把单次响应解析成与请求顺序对齐的 ``list[list[float]]``。

    body 类型签名故意保留 Any：上游服务偶发会返回 JSON ``null``（在我们对超长
    sentence 做边界截断后特别容易触发，原因是上游对"不完整结尾"的容错有差异），
    此时 ``resp.json()`` 给的就是 Python ``None``——这里在入口显式拦截，避免
    后续 ``body.get(...)`` 抛出含糊的 ``'NoneType' object has no attribute 'get'``。
    """

    if body is None:
        raise RuntimeError("embedding 响应体为 null（上游服务异常返回 JSON null）")
    if not isinstance(body, dict):
        raise RuntimeError(
            f"embedding 响应体类型异常 expected=dict got={type(body).__name__}"
        )

    if protocol == "openai":
        data = body.get("data")
        if not isinstance(data, list):
            raise RuntimeError(
                f"embedding(openai) 响应缺少 data 列表，实际 keys={list(body.keys())}"
            )
        if len(data) != batch_len:
            raise RuntimeError(
                f"embedding(openai) 响应数量不匹配 expected={batch_len} got={len(data)}"
            )
        placeholders: list[Optional[list[float]]] = [None] * batch_len
        for pos, item in enumerate(data):
            if not isinstance(item, dict):
                raise RuntimeError(f"data[{pos}] 不是 dict: {type(item).__name__}")
            emb = item.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError(
                    f"data[{pos}].embedding 类型异常: {type(emb).__name__}"
                )
            idx_raw = item.get("index")
            idx = int(idx_raw) if idx_raw is not None else pos
            if not (0 <= idx < batch_len):
                raise RuntimeError(f"data[{pos}].index={idx} 越界 batch_len={batch_len}")
            if placeholders[idx] is not None:
                raise RuntimeError(f"data 中 index={idx} 重复出现")
            placeholders[idx] = [float(x) for x in emb]
        if any(v is None for v in placeholders):
            missing = [i for i, v in enumerate(placeholders) if v is None]
            raise RuntimeError(f"embedding(openai) 响应缺失 index={missing}")
        return placeholders  # type: ignore[return-value]

    # BGE 协议分支（兼容旧服务）
    code = body.get("code")
    if code is not None and int(code) != 200:
        msg = str(body.get("msg") or "")
        err = f"embedding(bge) 服务返回失败 code={code} msg={msg!r}"
        # 已知硬性参数错误（如 sentence 超长、batch 超长）重试无意义，直接升级为
        # _NonRetryableEmbeddingError，让外层重试装饰器直接放行。
        if not _is_business_error_retryable(msg):
            raise _NonRetryableEmbeddingError(err)
        raise RuntimeError(err)
    output = body.get("output")
    if not isinstance(output, list):
        raise RuntimeError(
            f"embedding(bge) 响应缺少 output 列表，实际 keys={list(body.keys())}"
        )
    if len(output) != batch_len:
        raise RuntimeError(
            f"embedding(bge) 响应数量不匹配 expected={batch_len} got={len(output)}"
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
    model: Optional[str] = None,
    batch_size: int = _MAX_BATCH_SIZE,
    timeout_seconds: float = 60.0,
    client: Optional[httpx.AsyncClient] = None,
) -> list[list[float]]:
    """异步批量获取 embedding，输出顺序与 ``texts`` 保持一致。

    - 自动按 ``batch_size`` 分批，且按服务端硬限制夹到 ``_MAX_BATCH_SIZE`` (10) 以内；
    - 任何 HTTP 异常 / 业务 code 非 200 都直接 raise，调用方决定是否降级。
    """

    if not texts:
        return []
    effective_batch = _clamp_batch_size(batch_size)
    if effective_batch != batch_size:
        logger.info(
            "[Embedding] batch_size=%s 超过服务端上限，按 %d 切批",
            batch_size, effective_batch,
        )
    url, default_model, auth, protocol = _read_env()
    use_model = model or default_model
    headers = {"Content-Type": "application/json; charset=UTF-8"}
    if auth:
        headers["Authorization"] = auth

    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout_seconds, headers=headers)

    clipped_total = sum(1 for t in texts if len(t) > _MAX_INPUT_CHARS)
    if clipped_total:
        logger.info(
            "[Embedding] %d/%d 条文本超过 %d 字符上限，本次将对 embedding 副本做安全截断",
            clipped_total, len(texts), _MAX_INPUT_CHARS,
        )

    async def _do_one_request(batch: list[str]) -> list[list[float]]:
        """单批一次请求 + 解析；任何异常都抛出，由外层重试逻辑接管。"""
        payload = _build_payload(batch, use_model, protocol)
        resp = await cli.post(url, json=payload, headers=headers, timeout=timeout_seconds)
        if 400 <= resp.status_code < 500:
            raise _NonRetryableEmbeddingError(
                f"embedding HTTP {resp.status_code} body={resp.text[:500]}"
            )
        if resp.status_code >= 500:
            raise RuntimeError(
                f"embedding HTTP {resp.status_code} body={resp.text[:500]}"
            )
        try:
            body = resp.json()
        except Exception as e:
            raise RuntimeError(
                f"embedding 响应 JSON 解析失败: {e}; raw={resp.text[:500]!r}"
            ) from e
        return _parse_response(body, len(batch), protocol)

    async def _request_with_retry(batch: list[str], batch_idx: int) -> list[list[float]]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                return await _do_one_request(batch)
            except _NonRetryableEmbeddingError:
                raise
            except (httpx.RequestError, RuntimeError) as e:
                last_exc = e
                if attempt >= _RETRY_ATTEMPTS:
                    break
                delay = _retry_delay(attempt)
                logger.warning(
                    "[Embedding] batch_idx=%d 第 %d/%d 次请求失败，%.2fs 后重试: %s",
                    batch_idx, attempt, _RETRY_ATTEMPTS, delay, e,
                )
                await asyncio.sleep(delay)
        lens = [len(s) for s in batch]
        previews = [s[:80].replace("\n", " ") for s in batch]
        raise RuntimeError(
            f"{last_exc} | batch_idx={batch_idx} batch_size={len(batch)} "
            f"attempts={_RETRY_ATTEMPTS} lengths={lens} previews={previews}"
        ) from last_exc

    out: list[list[float]] = []
    try:
        for start in range(0, len(texts), effective_batch):
            batch_raw = texts[start : start + effective_batch]
            batch = [_clip_for_embedding(t) for t in batch_raw]
            batch_idx = start // effective_batch
            out.extend(await _request_with_retry(batch, batch_idx))
    finally:
        if own_client:
            await cli.aclose()

    logger.debug(
        "[Embedding] 取得 %d 条向量 (protocol=%s, model=%s, dim=%d)",
        len(out), protocol, use_model, len(out[0]) if out else 0,
    )
    return out


def embed_texts_sync(
    texts: list[str],
    *,
    model: Optional[str] = None,
    batch_size: int = _MAX_BATCH_SIZE,
    timeout_seconds: float = 60.0,
) -> list[list[float]]:
    """同步版本，便于离线脚本（如建索引 CLI）直接调用而不必绕 asyncio。

    与 :func:`embed_texts` 一样按服务端硬限制 ``_MAX_BATCH_SIZE`` (10) 夹断。
    """

    if not texts:
        return []
    import requests  # 延迟导入：异步路径不需要

    effective_batch = _clamp_batch_size(batch_size)
    if effective_batch != batch_size:
        logger.info(
            "[Embedding] (sync) batch_size=%s 超过服务端上限，按 %d 切批",
            batch_size, effective_batch,
        )
    url, default_model, auth, protocol = _read_env()
    use_model = model or default_model
    headers = {"Content-Type": "application/json; charset=UTF-8"}
    if auth:
        headers["Authorization"] = auth

    clipped_total = sum(1 for t in texts if len(t) > _MAX_INPUT_CHARS)
    if clipped_total:
        logger.info(
            "[Embedding] (sync) %d/%d 条文本超过 %d 字符上限，本次将对 embedding 副本做安全截断",
            clipped_total, len(texts), _MAX_INPUT_CHARS,
        )

    def _do_one_request(batch: list[str]) -> list[list[float]]:
        payload = _build_payload(batch, use_model, protocol)
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout_seconds)
        if 400 <= resp.status_code < 500:
            raise _NonRetryableEmbeddingError(
                f"embedding HTTP {resp.status_code} body={resp.text[:500]}"
            )
        if resp.status_code >= 500:
            raise RuntimeError(
                f"embedding HTTP {resp.status_code} body={resp.text[:500]}"
            )
        try:
            body = resp.json()
        except Exception as e:
            raise RuntimeError(
                f"embedding 响应 JSON 解析失败: {e}; raw={resp.text[:500]!r}"
            ) from e
        return _parse_response(body, len(batch), protocol)

    # 同步路径下网络错误类型来自 requests；与异步路径处理一致。
    _RETRYABLE_NET_EXC = (requests.exceptions.RequestException,)

    def _request_with_retry(batch: list[str], batch_idx: int) -> list[list[float]]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                return _do_one_request(batch)
            except _NonRetryableEmbeddingError:
                raise
            except (RuntimeError, *_RETRYABLE_NET_EXC) as e:
                last_exc = e
                if attempt >= _RETRY_ATTEMPTS:
                    break
                delay = _retry_delay(attempt)
                logger.warning(
                    "[Embedding] (sync) batch_idx=%d 第 %d/%d 次请求失败，%.2fs 后重试: %s",
                    batch_idx, attempt, _RETRY_ATTEMPTS, delay, e,
                )
                time.sleep(delay)
        lens = [len(s) for s in batch]
        previews = [s[:80].replace("\n", " ") for s in batch]
        raise RuntimeError(
            f"{last_exc} | batch_idx={batch_idx} batch_size={len(batch)} "
            f"attempts={_RETRY_ATTEMPTS} lengths={lens} previews={previews}"
        ) from last_exc

    out: list[list[float]] = []
    for start in range(0, len(texts), effective_batch):
        batch_raw = texts[start : start + effective_batch]
        batch = [_clip_for_embedding(t) for t in batch_raw]
        batch_idx = start // effective_batch
        out.extend(_request_with_retry(batch, batch_idx))

    logger.debug(
        "[Embedding] (sync) 取得 %d 条向量 (protocol=%s, model=%s, dim=%d)",
        len(out), protocol, use_model, len(out[0]) if out else 0,
    )
    return out

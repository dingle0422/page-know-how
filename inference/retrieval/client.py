"""retrieval_service 的官方 SDK：基于 httpx.AsyncClient 的异步 HTTP 客户端。

被以下三处调用：

- :mod:`inference.retrieval.indexer`：``build_for_root`` 把切好的 ``KnowledgeChunk`` +
  客户端 jieba 分词产物 + 客户端 embedding 一次性 upsert 到服务端。
- :mod:`inference.retrieval.hybrid`：``hybrid_search`` 用 query_tokenized + query_vector
  调 ``/search``。
- :mod:`app`：``_cascade_dependent_rebuilds`` 调 ``/v1/relations:lookup-dependents``
  反查依赖某 (policy_id, clause_id) 的所有源 policy。

设计要点：

- **单例 httpx.AsyncClient**：跨请求复用连接池，降低 5–15ms 的握手成本。失效时自动重建。
- **超时 + 重试**：写入与查询都跑短重试（指数退避 0.3/0.6/1.2s 三次）；HTTP 5xx 与网络错误重试，
  4xx 直接抛 ``RetrievalServiceError``，避免重试无意义。
- **不持有 LLM Key**：所有 embedding 仍由 ``inference.embedding_client`` 算好后送入。
- **降级策略**：服务不可达时 ``hybrid_search`` 应返回空列表（由调用方处理），不抛异常打断主流程；
  ``upsert`` 失败必须抛错（建索引失败要外显）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from typing import Any, Optional

import httpx

from knowledge_core.chunk_builder import KnowledgeChunk

logger = logging.getLogger(__name__)


# 默认配置（可被 :mod:`inference.config` 覆盖；这里直接读 env 是兜底，避免循环依赖）。
# 默认指向已部署的生产地址；本地起服务时设 RETRIEVAL_SERVICE_URL=http://127.0.0.1:8088 覆盖即可。
_DEFAULT_BASE_URL = os.getenv(
    "RETRIEVAL_SERVICE_URL", "http://mlp.paas.dc.servyou-it.com/kh-lancedb"
).strip().rstrip("/")
_DEFAULT_API_KEY = os.getenv("RETRIEVAL_SERVICE_API_KEY", "")
_DEFAULT_TIMEOUT = float(os.getenv("RETRIEVAL_SERVICE_TIMEOUT", "30.0"))
_RETRY_ATTEMPTS = max(1, int(os.getenv("RETRIEVAL_SERVICE_RETRY_ATTEMPTS", "3")))
_RETRY_BASE_DELAY = max(0.0, float(os.getenv("RETRIEVAL_SERVICE_RETRY_BASE_DELAY", "0.3")))
_RETRY_MAX_DELAY = max(_RETRY_BASE_DELAY, float(os.getenv("RETRIEVAL_SERVICE_RETRY_MAX_DELAY", "4.0")))


class RetrievalServiceError(RuntimeError):
    """服务端返回 4xx/5xx 或解析失败。"""


class RetrievalServiceUnavailable(RetrievalServiceError):
    """服务连不上（网络层）。``hybrid_search`` 等读路径会捕获此异常优雅降级到空结果。"""


_singleton_lock = asyncio.Lock()
_singleton: Optional["RetrievalServiceClient"] = None


async def get_default_client() -> "RetrievalServiceClient":
    """同进程内共享单例。"""

    global _singleton
    if _singleton is not None:
        return _singleton
    async with _singleton_lock:
        if _singleton is None:
            _singleton = RetrievalServiceClient()
        return _singleton


# ---------------------------------------------------------------- 序列化


def _kind_of(chunk: KnowledgeChunk) -> str:
    return "derived" if (chunk.parent_chunk_index and int(chunk.parent_chunk_index) > 0) else "original"


def _serialize_chunk_row(
    chunk: KnowledgeChunk,
    tokenized: str,
    vector: list[float],
) -> dict[str, Any]:
    """把一个 KnowledgeChunk + 客户端预算的 tokenized/vector 拍成服务端 ChunkRow dict。

    注意修复：旧版 ``inference/retrieval/indexer.py::_serialize_chunk`` 只序列化 4 个字段，
    导致派生 chunk 的 ``parent_chunk_index/derived_seq/relation_keys`` 在 jsonl 里全部丢失。
    本函数把这些字段补齐进 LanceDB schema，关联结构能被推理期 ``relations:expand`` /
    ``relations:lookup`` 反查命中。
    """

    rks = chunk.relation_keys or []
    relation_keys = [
        {"policy_id": (p or ""), "clause_id": (c or "")}
        for p, c in rks
        if isinstance(p, str) and isinstance(c, str)
    ]
    return {
        "chunk_id": int(chunk.index),
        "content": chunk.content or "",
        "content_tokenized": tokenized or "",
        "vector": list(vector or []),
        "heading_paths": [list(seg) for seg in (chunk.heading_paths or [])],
        "directories": [str(d) for d in (chunk.directories or [])],
        "kind": _kind_of(chunk),
        "parent_chunk_index": int(chunk.parent_chunk_index or -1),
        "derived_seq": int(chunk.derived_seq or 0),
        "relation_keys": relation_keys,
        "hop_depth": 0,  # KnowledgeChunk 当前未直接保存 hop_depth/source/clause_id
        "source": "",
        "clause_id": "",
        "built_at": 0,  # 服务端会自动补当前时间
    }


def _hit_to_knowledge_chunk(hit: dict) -> KnowledgeChunk:
    rks_raw = hit.get("relation_keys") or []
    relation_keys: list[tuple[str, str]] = [
        (rk.get("policy_id", ""), rk.get("clause_id", ""))
        for rk in rks_raw
        if isinstance(rk, dict)
    ]
    return KnowledgeChunk(
        index=int(hit.get("chunk_id", 0)),
        content=str(hit.get("content") or ""),
        heading_paths=list(hit.get("heading_paths") or []),
        directories=list(hit.get("directories") or []),
        parent_chunk_index=(
            int(hit["parent_chunk_index"])
            if hit.get("parent_chunk_index") not in (None, -1)
            else None
        ),
        derived_seq=int(hit.get("derived_seq") or 0),
        relation_keys=relation_keys,
    )


# ---------------------------------------------------------------- 客户端


class RetrievalServiceClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float | None = None,
    ) -> None:
        self._base_url = (base_url or _DEFAULT_BASE_URL).strip().rstrip("/")
        self._api_key = api_key if api_key is not None else _DEFAULT_API_KEY
        self._timeout = timeout if timeout is not None else _DEFAULT_TIMEOUT
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        return self._base_url

    async def _client_ref(self) -> httpx.AsyncClient:
        if self._client is not None and not self._client.is_closed:
            return self._client
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                headers = {}
                if self._api_key:
                    headers["X-API-Key"] = self._api_key
                self._client = httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=self._timeout,
                    headers=headers,
                    limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
                )
            return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    # ------------------------------------------------------------ 内部请求 + 重试

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        last_exc: Exception | None = None
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            client = await self._client_ref()
            try:
                resp = await client.request(method, path, **kwargs)
            except httpx.HTTPError as e:
                last_exc = e
                logger.warning(
                    "[RetrievalClient] %s %s 网络异常（attempt=%d/%d）: %s",
                    method, path, attempt, _RETRY_ATTEMPTS, e,
                )
                if attempt < _RETRY_ATTEMPTS:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                raise RetrievalServiceUnavailable(str(e)) from e

            status = resp.status_code
            if status >= 500:
                last_exc = RetrievalServiceError(
                    f"{method} {path} -> {status}: {resp.text[:200]}"
                )
                logger.warning(
                    "[RetrievalClient] %s %s -> %d（attempt=%d/%d）",
                    method, path, status, attempt, _RETRY_ATTEMPTS,
                )
                if attempt < _RETRY_ATTEMPTS:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                raise last_exc

            if status >= 400:
                # 4xx：参数/鉴权错误，重试无意义
                detail: Any = resp.text
                try:
                    detail = resp.json().get("detail", detail)
                except Exception:
                    pass
                raise RetrievalServiceError(f"{method} {path} -> {status}: {detail}")

            try:
                return resp.json()
            except (ValueError, json.JSONDecodeError) as e:
                raise RetrievalServiceError(
                    f"{method} {path}: 服务端返回非 JSON: {e}"
                ) from e

        # 不应到达
        if last_exc:
            raise last_exc
        raise RetrievalServiceError(f"{method} {path}: 未知失败")

    @staticmethod
    def _backoff(attempt: int) -> float:
        d = min(_RETRY_BASE_DELAY * (2 ** (attempt - 1)), _RETRY_MAX_DELAY)
        return d * (0.5 + random.random() * 0.5)  # jitter

    # ------------------------------------------------------------ 高级 API

    async def healthz(self) -> dict:
        return await self._request("GET", "/healthz")

    async def upsert_chunks(
        self,
        policy_id: str,
        rows: list[dict],
        *,
        mode: str = "overwrite",
        expected_dim: int | None = None,
    ) -> dict:
        body = {"chunks": rows, "mode": mode}
        if expected_dim is not None:
            body["expected_dim"] = int(expected_dim)
        return await self._request(
            "POST", f"/v1/policies/{_quote(policy_id)}/chunks:upsert", json=body
        )

    async def upsert_knowledge_chunks(
        self,
        policy_id: str,
        chunks: list[KnowledgeChunk],
        *,
        tokenized: list[str],
        vectors: list[list[float]],
        mode: str = "overwrite",
        expected_dim: int | None = None,
    ) -> dict:
        """高级写入：把 KnowledgeChunk 列表 + 客户端预算的 tokenized/vector 一次性发到服务端。

        ``vectors`` 可以为空列表（embedding 服务不可用时），此时服务端仅建 BM25。
        ``len(tokenized)`` 必须等于 ``len(chunks)``。
        """

        if len(tokenized) != len(chunks):
            raise ValueError(
                f"tokenized 长度与 chunks 不一致: {len(tokenized)} vs {len(chunks)}"
            )
        has_vec = bool(vectors)
        if has_vec and len(vectors) != len(chunks):
            raise ValueError(
                f"vectors 长度与 chunks 不一致: {len(vectors)} vs {len(chunks)}"
            )

        rows: list[dict] = []
        for i, c in enumerate(chunks):
            v = vectors[i] if has_vec else []
            rows.append(_serialize_chunk_row(c, tokenized[i], v))
        return await self.upsert_chunks(
            policy_id,
            rows,
            mode=mode,
            expected_dim=expected_dim,
        )

    async def search(
        self,
        policy_id: str,
        *,
        query_tokenized: str,
        query_vector: list[float],
        top_n: int = 20,
        top_m: int = 20,
        rrf_k: int | None = None,
        where: str | None = None,
        include_content: bool = True,
        include_derived: bool = True,
    ) -> list[KnowledgeChunk]:
        body: dict[str, Any] = {
            "query_tokenized": query_tokenized,
            "query_vector": list(query_vector or []),
            "top_n": int(top_n),
            "top_m": int(top_m),
            "include_content": bool(include_content),
            "include_derived": bool(include_derived),
        }
        if rrf_k is not None:
            body["rrf_k"] = int(rrf_k)
        if where:
            body["where"] = where
        try:
            data = await self._request(
                "POST", f"/v1/policies/{_quote(policy_id)}/search", json=body
            )
        except RetrievalServiceUnavailable as e:
            logger.warning("[RetrievalClient] search 服务不可达，降级到空结果: %s", e)
            return []
        hits = data.get("hits") or []
        return [_hit_to_knowledge_chunk(h) for h in hits]

    async def vector_search_v2(
        self,
        collection_id: str,
        *,
        query_vector: list[float],
        top_n: int,
        where: str | None = None,
        include_content: bool = True,
        include_derived: bool = True,
    ) -> list[dict]:
        """v2 纯向量检索：``POST /v2/collections/{collection_id}/search``。

        与 v1 ``/v1/policies/{}/search`` 的差异：

        - **走 v2 端点**：collection 命名空间独立（如 case_refinery 写入的
          ``case_{khCode}``），不与 inference 的 policy 索引混用；
        - **纯向量**：``query_tokenized=""``，``top_m=0``，仅走向量召回；
        - **暴露 cosine_similarity**：响应 ``hits[*]`` 内带原始 cosine 分，
          调用方可直接按阈值过滤（详见 :mod:`inference.retrieval.case_search`）。

        ``where``：LanceDB where 过滤表达式（如 ``md_case_polarity_xxx = 'positive'``）。
        **注意**：metadata 字段在服务端会被扁平化为 ``md_<path>_<hash8>`` 列，
        where 必须用该扁平化列名（用原始字段名 ``case_polarity`` 不报错但永远筛不到），
        列名以 :meth:`get_collection_meta_v2` 返回的 ``filterable_fields`` 为准。

        返回 raw ``hits`` list（每项含 ``document_id`` / ``score`` /
        ``cosine_similarity`` / ``content`` / ``metadata``），由调用方解析。

        失败降级：

        - 服务不可达 → ``[]`` （case 检索是 preview 的"锦上添花"，不应阻塞主流程）；
        - 4xx 含 ``404``（集合不存在）→ ``[]``（case_refinery 未上线是预期）；
        - 其他 4xx 仍抛 ``RetrievalServiceError`` 暴露调用方参数错误。
        """

        body: dict[str, Any] = {
            "query_tokenized": "",
            "query_vector": list(query_vector or []),
            "top_n": int(top_n),
            "top_m": 0,
            "include_content": bool(include_content),
            "include_derived": bool(include_derived),
            "strategy": "legacy_hybrid",
        }
        if where:
            body["where"] = where
        try:
            data = await self._request(
                "POST", f"/v2/collections/{_quote(collection_id)}/search", json=body,
            )
        except RetrievalServiceUnavailable as e:
            logger.warning(
                "[RetrievalClient] vector_search_v2 服务不可达 collection=%s: %s",
                collection_id, e,
            )
            return []
        except RetrievalServiceError as e:
            msg = str(e)
            if "404" in msg or "not found" in msg.lower() or "not indexed" in msg.lower():
                logger.info(
                    "[RetrievalClient] vector_search_v2 collection=%s 不存在/未建索引: %s",
                    collection_id, e,
                )
                return []
            raise
        return list(data.get("hits") or [])

    async def get_collection_meta_v2(self, collection_id: str) -> dict | None:
        """v2 集合元信息：``GET /v2/collections/{collection_id}/meta``。

        主要供 case 检索解析 ``filterable_fields`` 里的扁平化列名（如
        ``md_case_polarity_<hash8>`` / ``md_tombstoned_<hash8>``），用于构造 where。

        降级：服务不可达 / 集合不存在（404）→ ``None``，调用方据此跳过 where 筛选。
        """

        try:
            return await self._request(
                "GET", f"/v2/collections/{_quote(collection_id)}/meta"
            )
        except RetrievalServiceUnavailable as e:
            logger.warning(
                "[RetrievalClient] get_collection_meta_v2 服务不可达 collection=%s: %s",
                collection_id, e,
            )
            return None
        except RetrievalServiceError as e:
            if "404" in str(e):
                return None
            raise

    async def expand_relations(
        self,
        policy_id: str,
        chunk_id: int,
        *,
        include_content: bool = True,
    ) -> list[KnowledgeChunk]:
        body = {"chunk_id": int(chunk_id), "include_content": bool(include_content)}
        try:
            data = await self._request(
                "POST",
                f"/v1/policies/{_quote(policy_id)}/relations:expand",
                json=body,
            )
        except RetrievalServiceUnavailable as e:
            logger.warning("[RetrievalClient] expand 服务不可达: %s", e)
            return []
        return [_hit_to_knowledge_chunk(h) for h in (data.get("chunks") or [])]

    async def lookup_in_policy(
        self,
        policy_id: str,
        *,
        target_policy_id: str,
        target_clause_id: str | None = None,
        include_content: bool = False,
    ) -> list[KnowledgeChunk]:
        params = {
            "target_policy_id": target_policy_id,
            "include_content": "true" if include_content else "false",
        }
        if target_clause_id:
            params["target_clause_id"] = target_clause_id
        try:
            data = await self._request(
                "GET",
                f"/v1/policies/{_quote(policy_id)}/relations:lookup",
                params=params,
            )
        except RetrievalServiceUnavailable as e:
            logger.warning("[RetrievalClient] lookup_in_policy 服务不可达: %s", e)
            return []
        return [_hit_to_knowledge_chunk(h) for h in (data.get("chunks") or [])]

    async def lookup_dependents(
        self,
        target_policy_id: str,
        target_clause_id: str | None = None,
    ) -> list[dict]:
        """``[{source_policy_id, n_hits}, ...]``。"""

        params: dict[str, str] = {"target_policy_id": target_policy_id}
        if target_clause_id:
            params["target_clause_id"] = target_clause_id
        try:
            data = await self._request(
                "GET", "/v1/relations:lookup-dependents", params=params
            )
        except RetrievalServiceUnavailable as e:
            logger.warning("[RetrievalClient] lookup_dependents 服务不可达: %s", e)
            return []
        return list(data.get("dependents") or [])

    async def list_chunks(
        self,
        policy_id: str,
        *,
        where: str | None = None,
        limit: int = 1000,
        include_content: bool = False,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "limit": int(limit),
            "include_content": "true" if include_content else "false",
        }
        if where:
            params["where"] = where
        try:
            return await self._request(
                "GET", f"/v1/policies/{_quote(policy_id)}/chunks", params=params
            )
        except RetrievalServiceUnavailable as e:
            logger.warning("[RetrievalClient] list_chunks 服务不可达: %s", e)
            return []

    async def get_meta(self, policy_id: str) -> dict | None:
        try:
            return await self._request(
                "GET", f"/v1/policies/{_quote(policy_id)}/meta"
            )
        except RetrievalServiceError as e:
            # 404 也走这里
            if "404" in str(e):
                return None
            raise

    async def list_policies(self) -> list[dict]:
        try:
            data = await self._request("GET", "/v1/policies")
        except RetrievalServiceUnavailable as e:
            logger.warning("[RetrievalClient] list_policies 服务不可达: %s", e)
            return []
        return list(data.get("policies") or [])

    async def drop_policy(self, policy_id: str) -> bool:
        data = await self._request(
            "DELETE", f"/v1/policies/{_quote(policy_id)}"
        )
        return bool(data.get("ok"))


def _quote(s: str) -> str:
    """policy_id 含中文/特殊字符时安全 URL 编码。"""

    from urllib.parse import quote

    return quote(s or "", safe="")

"""hybrid_search：BM25 + 语义 + RRF 的统一入口（HTTP 版）。

迁移到 ``retrieval_service`` 后，本模块对外签名保持不变：

    chunks = await hybrid_search(question, policy_id, top_n=20, top_m=20)

返回 ``list[KnowledgeChunk]``，可直接喂给 :func:`inference.react_loop.run`。

内部职责仅剩：

1. 客户端 jieba 分词得到 ``query_tokenized``（与服务端 FTS 完全同源）；
2. 调 :func:`inference.embedding_client.embed_texts` 得到 ``query_vector``；
3. 调 :class:`inference.retrieval.client.RetrievalServiceClient.search` 走 HTTP 拿命中；
4. 服务端不可达 / embedding 缺配置时优雅降级（部分通路缺失即少传 query_vector / query_tokenized）。

服务端 ``store.hybrid_search`` 内部已经做了 RRF 融合，公式与原 :mod:`inference.retrieval.rrf`
一致（``1/(k+rank)``，``k=60``）。
"""

from __future__ import annotations

import asyncio
import logging
import time

from reasoner.v3.chunk_builder import KnowledgeChunk

from . import bm25 as bm25_mod
from .client import RetrievalServiceUnavailable, get_default_client

logger = logging.getLogger(__name__)


# 自愈：当 embedding 服务暂时不可用时，hybrid_search 仍能返回 BM25-only 结果；
# 后台异步触发一次 :func:`indexer.rebuild_embeddings_only`，5 min 冷却避免空打。
_REBUILD_INFLIGHT: set[str] = set()
_REBUILD_LAST_ATTEMPT: dict[str, float] = {}
_REBUILD_LOCK = asyncio.Lock()
_REBUILD_COOLDOWN_SECONDS: float = 300.0  # 5 min


def invalidate(policy_id: str | None = None) -> None:
    """老接口保留：retrieval_service 化后没有进程内表缓存需要清，仅用于 trace。

    保留是为了 :mod:`app` 在重建索引完成后 ``_invalidate_hybrid_cache(policy_id)`` 调用兼容。
    """

    if policy_id:
        logger.debug("[Hybrid] invalidate(policy_id=%s) noop（已迁移到服务端）", policy_id)
    else:
        logger.debug("[Hybrid] invalidate(all) noop（已迁移到服务端）")


# ---------------------------------------------------------------- 主入口


async def _query_vector(question: str) -> list[float]:
    """拿单条 query 向量；embedding 服务异常时返回 ``[]``，由调用方走纯 BM25。"""

    try:
        from ..embedding_client import EmbeddingNotConfigured, embed_texts
    except Exception as e:  # pragma: no cover
        logger.warning("[Hybrid] 导入 embedding_client 失败: %s", e)
        return []

    try:
        vecs = await embed_texts([question])
    except EmbeddingNotConfigured as e:
        logger.info("[Hybrid] embedding 未配置，走纯 BM25: %s", e)
        return []
    except Exception as e:
        logger.warning("[Hybrid] 取 query 向量失败，走纯 BM25: %s", e)
        return []
    if not vecs:
        return []
    return list(vecs[0] or [])


async def _maybe_trigger_embedding_rebuild(policy_id: str) -> None:
    """fire-and-forget：本次查询发现服务端无向量列时触发后台重算 embedding 列。

    复用 ``inference.retrieval.indexer.rebuild_embeddings_only``：拉服务端 chunks → 客户端 embed →
    merge 回写 vector 列。失败时刷 cooldown 避免对失败的 embedding 服务死循环重试。
    """

    async with _REBUILD_LOCK:
        if policy_id in _REBUILD_INFLIGHT:
            return
        last = _REBUILD_LAST_ATTEMPT.get(policy_id)
        if last is not None and (time.monotonic() - last) < _REBUILD_COOLDOWN_SECONDS:
            return
        _REBUILD_INFLIGHT.add(policy_id)

    async def _run() -> None:
        try:
            from .indexer import rebuild_embeddings_only

            logger.info("[Hybrid] 后台重算 embedding 列：policy_id=%s", policy_id)
            result = await rebuild_embeddings_only(policy_id)
            logger.info("[Hybrid] 后台重算完成 policy_id=%s: %s", policy_id, result)
        except Exception as e:
            logger.warning("[Hybrid] 后台重算异常 policy_id=%s: %s", policy_id, e)
        finally:
            _REBUILD_LAST_ATTEMPT[policy_id] = time.monotonic()
            _REBUILD_INFLIGHT.discard(policy_id)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run())
    except RuntimeError:
        _REBUILD_INFLIGHT.discard(policy_id)


async def hybrid_search(
    question: str,
    policy_id: str,
    *,
    top_n: int = 20,
    top_m: int = 20,
    rrf_k: int | None = None,
) -> list[KnowledgeChunk]:
    """混合检索主入口，详见模块 docstring。"""

    if not question or not question.strip():
        return []
    if not policy_id:
        return []

    # 1) 客户端分词（与服务端 FTS 100% 同源）
    q_tok = bm25_mod.tokenize_join(question)

    # 2) 客户端取 query 向量
    q_vec = await _query_vector(question)

    if not q_tok and not q_vec:
        logger.warning("[Hybrid] policy_id=%s 分词与向量均为空，返回空", policy_id)
        return []

    # 3) HTTP 调服务端检索
    try:
        client = await get_default_client()
        chunks = await client.search(
            policy_id,
            query_tokenized=q_tok,
            query_vector=q_vec,
            top_n=top_n,
            top_m=top_m,
            rrf_k=rrf_k,
            include_content=True,
            include_derived=True,
        )
    except RetrievalServiceUnavailable as e:
        logger.warning(
            "[Hybrid] retrieval_service 不可达，policy_id=%s 返回空: %s", policy_id, e
        )
        return []
    except Exception as e:
        logger.warning(
            "[Hybrid] 调 retrieval_service 失败 policy_id=%s: %s", policy_id, e
        )
        return []

    # 4) 自愈：服务端有数据但向量缺失（dim=0）→ 后台触发重建
    if not q_vec:
        # query 向量自身就空；判断服务端是否也无向量列
        try:
            client = await get_default_client()
            meta = await client.get_meta(policy_id)
            if meta and not int(meta.get("dim", 0)):
                await _maybe_trigger_embedding_rebuild(policy_id)
        except Exception:
            pass

    return chunks

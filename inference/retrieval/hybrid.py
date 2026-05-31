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
from .client import RetrievalServiceError, RetrievalServiceUnavailable, get_default_client

logger = logging.getLogger(__name__)


# 自愈：当 embedding 服务暂时不可用时，hybrid_search 仍能返回 BM25-only 结果；
# 后台异步触发一次 :func:`indexer.rebuild_embeddings_only`，5 min 冷却避免空打。
_REBUILD_INFLIGHT: set[str] = set()
_REBUILD_LAST_ATTEMPT: dict[str, float] = {}
_REBUILD_LOCK = asyncio.Lock()
_REBUILD_COOLDOWN_SECONDS: float = 300.0  # 5 min


# 情况 B 自愈：本地 page_knowledge 仍在、但服务端 LanceDB 表整张丢失（搜索 404
# "policy not indexed"）。这种不一致 _inference_artifacts_stale 探测不到（它只看本地
# meta），不会自动重建，于是会稳定 404 + 空召回。这里在检索侧观测到 404 时，
# 直接基于本地目录 build_for_root 重新在服务端建表 + upsert，再重试一次检索。
#
# - per-policy 锁：同一索引串行重建，并发请求只会触发一次 build；
# - 失败冷却：重建失败 / 重建后仍无表（如本地 knowledge 为空）时刷冷却时间戳，
#   避免对坏目录或坏服务端在每次查询上死循环重建。
_INDEX_REBUILD_LOCKS: dict[str, asyncio.Lock] = {}
_INDEX_REBUILD_LOCKS_GUARD = asyncio.Lock()
_INDEX_REBUILD_LAST_ATTEMPT: dict[str, float] = {}
_INDEX_REBUILD_COOLDOWN_SECONDS: float = 300.0  # 5 min


def _is_not_indexed_error(exc: Exception) -> bool:
    """判断异常是否为服务端"该 policy 未建索引"（表不存在）。

    服务端对未建索引的 policy 返回 ``404: policy not indexed: ...``，经
    :class:`RetrievalServiceClient` 包装为 ``RetrievalServiceError`` 抛出。
    """

    msg = str(exc).lower()
    return "not indexed" in msg or "not found" in msg or "404" in msg


async def _get_index_rebuild_lock(key: str) -> asyncio.Lock:
    async with _INDEX_REBUILD_LOCKS_GUARD:
        lock = _INDEX_REBUILD_LOCKS.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _INDEX_REBUILD_LOCKS[key] = lock
        return lock


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


async def _search_once(
    index_policy_id: str,
    *,
    query_tokenized: str,
    query_vector: list[float],
    top_n: int,
    top_m: int,
    rrf_k: int | None,
) -> list[KnowledgeChunk]:
    """单次服务端检索（不含自愈），供 :func:`hybrid_search` 与自愈重试复用。"""

    client = await get_default_client()
    return await client.search(
        index_policy_id,
        query_tokenized=query_tokenized,
        query_vector=query_vector,
        top_n=top_n,
        top_m=top_m,
        rrf_k=rrf_k,
        include_content=True,
        include_derived=True,
    )


async def _recover_missing_index(
    index_policy_id: str,
    *,
    query_tokenized: str,
    query_vector: list[float],
    top_n: int,
    top_m: int,
    rrf_k: int | None,
) -> list[KnowledgeChunk] | None:
    """情况 B 自愈：服务端表丢失但本地 page_knowledge 仍在 → 重建后重试检索。

    流程：

    1. 由 ``index_policy_id``（``{base}__cs{N}``）拆出基础 policy_id + chunk_size；
    2. ``resolve_root_dir`` 映射回本地 ``page_knowledge/<dir>``，不存在则放弃（无米下锅）；
    3. per-policy 锁内先重试一次检索（并发的另一个请求可能已重建好）；
    4. 仍 404 且未在冷却期 → ``build_for_root`` 基于本地目录重新切块/分词/embedding 并
       整表 upsert 到服务端，重建该 collection；
    5. 重建成功后重试一次检索并返回结果。

    返回：

    - ``list``：重建 + 重试成功（空 list 表示表已就绪但本次查询无命中）；
    - ``None``：无法自愈（本地目录缺失 / 冷却中 / 重建失败 / 重建后仍无表），
      调用方据此降级到空结果。
    """

    from .. import config as _cfg
    from .indexer import build_for_root, parse_index_policy_id, resolve_root_dir

    base_pid, chunk_size = parse_index_policy_id(index_policy_id)
    if chunk_size is None:
        chunk_size = int(_cfg.INFERENCE_DEFAULT_CHUNK_SIZE)

    root_dir = resolve_root_dir(base_pid)
    if not root_dir:
        logger.warning(
            "[Hybrid] 自愈失败：policy_id=%s 在本地 page_knowledge 找不到对应目录，无法重建",
            index_policy_id,
        )
        return None

    lock = await _get_index_rebuild_lock(index_policy_id)
    async with lock:
        # 进锁后先重试一次：并发请求可能已经把表重建好，避免重复 build。
        try:
            return await _search_once(
                index_policy_id,
                query_tokenized=query_tokenized,
                query_vector=query_vector,
                top_n=top_n,
                top_m=top_m,
                rrf_k=rrf_k,
            )
        except RetrievalServiceError as e:
            if not _is_not_indexed_error(e):
                # 其它错误（鉴权/参数/5xx）非本自愈职责，交回调用方降级。
                logger.warning(
                    "[Hybrid] 自愈前重试检索遇到非 not-indexed 错误 policy_id=%s: %s",
                    index_policy_id, e,
                )
                return None
            # 仍是 not indexed：确实需要本任务重建。

        last = _INDEX_REBUILD_LAST_ATTEMPT.get(index_policy_id)
        if last is not None and (time.monotonic() - last) < _INDEX_REBUILD_COOLDOWN_SECONDS:
            logger.info(
                "[Hybrid] 自愈冷却中 policy_id=%s（上次重建 %.0fs 前），本次降级空结果",
                index_policy_id, time.monotonic() - last,
            )
            return None

        logger.info(
            "[Hybrid] 自愈：服务端无索引，基于本地 page_knowledge 重建 "
            "policy_id=%s root=%s chunk_size=%d",
            index_policy_id, root_dir, chunk_size,
        )
        try:
            result = await build_for_root(
                root_dir,
                policy_id=base_pid,
                chunk_size=chunk_size,
            )
            logger.info("[Hybrid] 自愈重建完成 policy_id=%s: %s", index_policy_id, result)
        except Exception as e:
            _INDEX_REBUILD_LAST_ATTEMPT[index_policy_id] = time.monotonic()
            logger.warning(
                "[Hybrid] 自愈重建失败 policy_id=%s: %s", index_policy_id, e, exc_info=True
            )
            return None

        # 重建成功后重试检索；仍异常（如本地 knowledge 为空导致服务端无表）→ 刷冷却避免死循环。
        try:
            chunks = await _search_once(
                index_policy_id,
                query_tokenized=query_tokenized,
                query_vector=query_vector,
                top_n=top_n,
                top_m=top_m,
                rrf_k=rrf_k,
            )
            _INDEX_REBUILD_LAST_ATTEMPT.pop(index_policy_id, None)
            return chunks
        except Exception as e:
            _INDEX_REBUILD_LAST_ATTEMPT[index_policy_id] = time.monotonic()
            logger.warning(
                "[Hybrid] 自愈重建后重试检索仍失败 policy_id=%s: %s", index_policy_id, e
            )
            return None


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
        chunks = await _search_once(
            policy_id,
            query_tokenized=q_tok,
            query_vector=q_vec,
            top_n=top_n,
            top_m=top_m,
            rrf_k=rrf_k,
        )
    except RetrievalServiceUnavailable as e:
        logger.warning(
            "[Hybrid] retrieval_service 不可达，policy_id=%s 返回空: %s", policy_id, e
        )
        return []
    except RetrievalServiceError as e:
        # 情况 B：服务端表丢失但本地 page_knowledge 仍在 → 基于本地重建后重试检索。
        if _is_not_indexed_error(e):
            logger.warning(
                "[Hybrid] 服务端无索引 policy_id=%s，尝试基于本地 page_knowledge 自愈重建: %s",
                policy_id, e,
            )
            recovered = await _recover_missing_index(
                policy_id,
                query_tokenized=q_tok,
                query_vector=q_vec,
                top_n=top_n,
                top_m=top_m,
                rrf_k=rrf_k,
            )
            return recovered if recovered is not None else []
        logger.warning(
            "[Hybrid] 调 retrieval_service 失败 policy_id=%s: %s", policy_id, e
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

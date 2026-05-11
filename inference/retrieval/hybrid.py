"""hybrid_search：BM25 + 语义 + RRF 的统一入口。

外部仅需要：

    chunks = await hybrid_search(question, policy_id, top_n=20, top_m=20)

返回 ``list[KnowledgeChunk]``，可直接喂给 :func:`inference.react_loop.run`。

策略：

1. ``policy_id -> knowledge_root``：通过 ``extractor/policy_index.get_root_map`` 解析。
2. 进程内 LRU 缓存按 ``policy_id`` 懒加载三件套（chunks / bm25 / embeddings）。
3. BM25 取 top_m，语义取 top_n；
4. RRF 融合后回填为 ``KnowledgeChunk`` 列表（按融合分数排序），
   返回上限取 ``top_n`` 与 ``top_m`` 两路候选的去重数量（而非 ``max(top_n, top_m)``）。
5. 索引缺失/服务不可用时回退到纯 BM25（甚至空列表），不抛异常。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from reasoner.v3.chunk_builder import KnowledgeChunk

from . import bm25 as bm25_mod
from . import rrf as rrf_mod
from . import semantic as semantic_mod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 路径解析

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PAGE_KNOWLEDGE_DIR = os.path.join(_PROJECT_ROOT, "page_knowledge")
_POLICY_INDEX_FILE = os.path.join(_PAGE_KNOWLEDGE_DIR, "_policy_index.json")


def _resolve_root_dir(policy_id: str) -> Optional[str]:
    from extractor.policy_index import get_root_map

    root_map = get_root_map(_POLICY_INDEX_FILE)
    rel = root_map.get(policy_id)
    if not rel:
        return None
    abs_dir = os.path.join(_PAGE_KNOWLEDGE_DIR, rel)
    return abs_dir if os.path.isdir(abs_dir) else None


def index_paths(root_dir: str) -> dict[str, str]:
    return {
        "chunks": os.path.join(root_dir, "_chunks.jsonl"),
        "bm25": os.path.join(root_dir, "_bm25.pkl"),
        "embeddings": os.path.join(root_dir, "_embeddings.npy"),
    }


# ---------------------------------------------------------------- 三件套加载

@dataclass
class _PolicyArtifacts:
    root_dir: str
    chunks: list[KnowledgeChunk]
    bm25_index: dict | None
    embeddings: object | None  # numpy ndarray or None


_CACHE: dict[str, _PolicyArtifacts] = {}
_CACHE_LOCK = asyncio.Lock()


def _read_chunks_jsonl(path: str) -> list[KnowledgeChunk]:
    if not os.path.exists(path):
        return []
    out: list[KnowledgeChunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                out.append(KnowledgeChunk(
                    index=int(obj.get("index") or len(out) + 1),
                    content=str(obj.get("content") or ""),
                    heading_paths=list(obj.get("heading_paths") or []),
                    directories=list(obj.get("directories") or []),
                ))
            except Exception as e:
                logger.warning("[Hybrid] 跳过非法 chunk 行: %s", e)
    return out


async def _load_artifacts(policy_id: str) -> Optional[_PolicyArtifacts]:
    cached = _CACHE.get(policy_id)
    if cached is not None:
        return cached
    async with _CACHE_LOCK:
        cached = _CACHE.get(policy_id)
        if cached is not None:
            return cached
        root_dir = _resolve_root_dir(policy_id)
        if not root_dir:
            logger.warning("[Hybrid] policy_id=%s 未找到知识根目录", policy_id)
            return None

        paths = index_paths(root_dir)
        chunks = await asyncio.to_thread(_read_chunks_jsonl, paths["chunks"])
        if not chunks:
            logger.warning(
                "[Hybrid] policy_id=%s 缺少 _chunks.jsonl，回退到空检索结果", policy_id
            )
            return None
        bm25_index = await asyncio.to_thread(bm25_mod.load, paths["bm25"])
        embeddings = await asyncio.to_thread(semantic_mod.load, paths["embeddings"])

        artifacts = _PolicyArtifacts(
            root_dir=root_dir,
            chunks=chunks,
            bm25_index=bm25_index,
            embeddings=embeddings,
        )
        _CACHE[policy_id] = artifacts
        return artifacts


def invalidate(policy_id: Optional[str] = None) -> None:
    """显式失效缓存（重新建索引后调用）。"""

    if policy_id is None:
        _CACHE.clear()
    else:
        _CACHE.pop(policy_id, None)


# ---------------------------------------------------------------- 主入口


async def hybrid_search(
    question: str,
    policy_id: str,
    *,
    top_n: int = 20,
    top_m: int = 20,
    rrf_k: int | None = None,
) -> list[KnowledgeChunk]:
    """混合检索主入口，详见模块 docstring。"""

    if not question.strip():
        return []
    artifacts = await _load_artifacts(policy_id)
    if artifacts is None:
        return []

    chunks = artifacts.chunks
    n_chunks = len(chunks)

    # BM25 路径（同步、CPU 密集，丢线程池）
    bm25_pairs: list[tuple[int, float]] = []
    if artifacts.bm25_index is not None:
        bm25_pairs = await asyncio.to_thread(
            bm25_mod.search, artifacts.bm25_index, question, top_m
        )
    else:
        logger.warning(
            "[Hybrid] policy_id=%s 缺少 BM25 索引；"
            "将仅基于 semantic（如可用）", policy_id
        )

    # 语义路径
    sem_pairs: list[tuple[int, float]] = []
    if artifacts.embeddings is not None:
        sem_pairs = await semantic_mod.search(artifacts.embeddings, question, top_n)
    else:
        if bm25_pairs:
            logger.info("[Hybrid] policy_id=%s 缺少向量库，回退到纯 BM25", policy_id)

    if not bm25_pairs and not sem_pairs:
        return []

    # 返回上限：两路候选合并后的去重数量（<= top_n + top_m）。
    dedup_limit = len({
        *(cid for cid, _ in bm25_pairs),
        *(cid for cid, _ in sem_pairs),
    })

    fused = rrf_mod.reciprocal_rank_fusion(
        [bm25_pairs, sem_pairs],
        k=rrf_k if rrf_k is not None else 60,
        top_k=dedup_limit,
    )

    out: list[KnowledgeChunk] = []
    for cid, _score in fused:
        if 0 <= cid < n_chunks:
            out.append(chunks[cid])
    return out

"""离线建索引：扫 ``page_knowledge/{root}/`` 切块 + 高亮外链派生 chunk + BM25 + embedding 落盘。

入口 :func:`build_for_root`：

1. 调 ``reasoner/v3/chunk_builder.build_knowledge_chunks(root, chunk_size)`` 切块；
2. 复用 ``reasoner/v3/relation_crawler.RelationCrawler``（``expand_all=True``，
   纯静态展开、不走 LLM）扫该 root 下所有 ``clause.json`` 的外链 ``highlightedContent``
   引用图，多跳 BFS 抓回关联条款，渲染成派生 ``KnowledgeChunk`` 与原始 chunks 一并入索引；
3. 写 ``_chunks.jsonl``；
4. 用 :mod:`inference.retrieval.bm25` 建 BM25 并写 ``_bm25.pkl``；
5. 调 :func:`inference.embedding_client.embed_texts` 批量取向量并写 ``_embeddings.npy``，
   embedding 服务未配置时跳过该步骤（hybrid 检索会自动退化为纯 BM25）。
6. 写 ``_index_meta.json`` 与 ``_relation_targets.json``，分别记录 schema 版本+chunk 计数
   + stale 标记，以及本 root 派生 chunks 依赖的外部 (policy_id, clause_id)。后者是
   ``app.py._cascade_dependent_rebuilds`` 做跨 policy 级联刷新的反向追踪表。

幂等：每次都整文件覆盖，不增量。
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from reasoner._registries import RelationFragment, RelationRegistry
from reasoner.v3.chunk_builder import (
    KnowledgeChunk,
    build_knowledge_chunks,
    split_relations_into_chunks,
)
from reasoner.v3.clause_locator import ClauseLocator
from reasoner.v3.relation_crawler import RelationCrawler

from .. import config
from . import bm25 as bm25_mod
from . import semantic as semantic_mod

logger = logging.getLogger(__name__)


def _serialize_chunk(c: KnowledgeChunk) -> dict:
    return {
        "index": c.index,
        "content": c.content,
        "heading_paths": list(c.heading_paths or []),
        "directories": list(c.directories or []),
    }


def _write_chunks_jsonl(path: str, chunks: list[KnowledgeChunk]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(_serialize_chunk(c), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _write_json_atomic(path: str, payload: dict) -> None:
    """整文件原子写小 JSON，失败不抛（meta/targets 文件丢失只会让兜底重建路径再跑一次）。"""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("[Indexer] 写 %s 失败（忽略）: %s", path, e)


def _collect_relation_chunks(
    root_dir: str,
    original_chunks: list[KnowledgeChunk],
    *,
    chunk_size: int,
    max_depth: int,
    max_nodes: int,
    allow_remote: bool,
    remote_timeout: float,
) -> tuple[list[KnowledgeChunk], list[dict]]:
    """扫 root_dir 下所有 clause.json，静态展开高亮外链 -> 派生 KnowledgeChunk。

    与 reasoner 在线模式（HighlightPrecheck + RelationCrawler）共用同一套 crawler，
    但这里 ``expand_all=True`` + ``question=""``：跳过所有 LLM 判定，定位到内容即入
    fragment。本函数**不修改** original_chunks，但派生 chunk 的 ``parent_chunk_index``
    会指向最近的原始 chunk（按 source_dir 反查），便于 trace；定位不到时填 -1。

    返回 (derived_chunks, relation_targets)：

    - derived_chunks：已分配最终 index（从 ``len(original_chunks)+1`` 起递增），
      可直接 ``extend`` 进 chunks 列表后写入 ``_chunks.jsonl``。
    - relation_targets：``[{"policy_id": ..., "clause_id": ...}, ...]`` 的反向追踪表，
      供 ``_cascade_dependent_rebuilds`` 在某 policy 更新时找出哪些 root 需要重建。
    """
    page_knowledge_dir = os.path.dirname(os.path.abspath(root_dir))
    policy_index_path = os.path.join(page_knowledge_dir, "_policy_index.json")

    locator = ClauseLocator(
        page_knowledge_dir=page_knowledge_dir,
        policy_index_path=policy_index_path,
        remote_timeout=remote_timeout,
    )
    if not allow_remote:
        # 强制关闭远程兜底：替换 _try_remote 为返回 None 的桩，避免每条都打一次失败请求。
        # locator 的 cache 仍按 (policy_id, clause_id) memo，"local miss" 同样会被缓存。
        locator._try_remote = lambda _pid, _cid: None  # type: ignore[assignment]

    registry = RelationRegistry()
    # 把 dir_abspath → 原始 chunk 反查表先建好，给派生 chunk 填 parent_chunk_index 用。
    # original_chunks 的 directories 是该 chunk 涵盖的所有目录绝对路径，反查时按精确匹配。
    dir_to_chunk_index: dict[str, int] = {}
    for c in original_chunks:
        for d in c.directories or []:
            dir_to_chunk_index.setdefault(os.path.abspath(d), c.index)

    # 与 reasoner 在线路径同源的线程池；离线索引下并发度小一点足够，避免大量并发远程拉取。
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="idx-relcrawl") as executor:
        crawler = RelationCrawler(
            question="",
            registry=registry,
            locator=locator,
            executor=executor,
            max_depth=max_depth,
            max_nodes=max_nodes,
            expand_all=True,
        )

        n_dirs_scanned = 0
        for cur_root, _dirs, files in os.walk(root_dir):
            if "clause.json" not in files:
                continue
            n_dirs_scanned += 1
            source_dir_abs = os.path.abspath(cur_root)
            source_chunk_index = dir_to_chunk_index.get(source_dir_abs, -1)
            try:
                crawler.crawl(
                    source_chunk_index=source_chunk_index,
                    source_dir=source_dir_abs,
                    parent_assessment="离线索引构建：静态展开高亮外链（无 LLM 判定）",
                )
            except Exception as e:
                logger.warning(
                    "[Indexer] 关联展开失败 source_dir=%s: %s", source_dir_abs, e
                )

    all_fragments: list[RelationFragment] = registry.get_all()
    if not all_fragments:
        logger.info(
            "[Indexer] 扫描 %d 个含 clause.json 的目录，未抓到任何关联条款",
            n_dirs_scanned,
        )
        return [], []

    # 把 fragments 按其父 chunk index 分桶，逐桶喂给 split_relations_into_chunks。
    # 桶内顺序保留 registry 的 BFS 命中顺序，桶间按原始 chunk index 升序。
    by_parent: dict[int, list[RelationFragment]] = {}
    for frag in all_fragments:
        by_parent.setdefault(frag.parent_chunk_index, []).append(frag)

    # 给"父 chunk 没找到"的孤儿挂一个虚拟父 chunk，让派生 chunk 仍能产出但不归属任何原始 dir。
    orphan_parent = KnowledgeChunk(
        index=0,
        content="",
        heading_paths=[],
        directories=[],
    )
    original_by_index: dict[int, KnowledgeChunk] = {c.index: c for c in original_chunks}

    derived_chunks: list[KnowledgeChunk] = []
    next_index = len(original_chunks) + 1
    derived_seq_per_parent: dict[int, int] = {}

    for parent_idx in sorted(by_parent.keys()):
        fragments = by_parent[parent_idx]
        parent_chunk = original_by_index.get(parent_idx, orphan_parent)
        start_seq = derived_seq_per_parent.get(parent_idx, 0) + 1
        derived = split_relations_into_chunks(
            fragments,
            chunk_size=chunk_size,
            parent_chunk=parent_chunk,
            start_derived_seq=start_seq,
            knowledge_root=root_dir,
        )
        if not derived:
            continue
        # split 内部 index=0 占位，这里统一分配最终 index 并累计 seq。
        for dc in derived:
            dc.index = next_index
            next_index += 1
            derived_chunks.append(dc)
        derived_seq_per_parent[parent_idx] = start_seq + len(derived) - 1

    # 派生 chunks 之间允许 (policy_id, clause_id) 重复出现（一个目标条款可能跨多个父 chunk 被引用），
    # 但反向追踪表用于 cascade 触发器，按 policy_id 去重即可。
    seen_target_keys: set[tuple[str, str]] = set()
    relation_targets: list[dict] = []
    for frag in all_fragments:
        key = (frag.policy_id, frag.clause_id)
        if key in seen_target_keys:
            continue
        seen_target_keys.add(key)
        relation_targets.append({
            "policy_id": frag.policy_id,
            "clause_id": frag.clause_id,
            "source": frag.source,
        })

    logger.info(
        "[Indexer] 关联展开完成：%d 个父章节扫描，命中 %d 个唯一关联条款，产出 %d 个派生 chunk",
        n_dirs_scanned, len(all_fragments), len(derived_chunks),
    )
    return derived_chunks, relation_targets


async def build_for_root(
    root_dir: str,
    *,
    chunk_size: int = config.CHUNK_SIZE,
    # 服务端单次最多接收 10 条；这里默认就贴着上限，避免 embedding_client 内
    # 二次夹断时再打一条 INFO。如有更小限制可手工传入更小值。
    embedding_batch_size: int = 10,
    embedding_model: Optional[str] = None,
    skip_embedding: bool = False,
    include_relations: Optional[bool] = None,
    relation_max_depth: Optional[int] = None,
    relation_max_nodes: Optional[int] = None,
    relation_allow_remote: Optional[bool] = None,
    relation_remote_timeout: Optional[float] = None,
) -> dict:
    """对单个知识根目录建全套索引，返回 ``{"chunks": N, "bm25": bool, "embeddings": bool,
    "n_original": N1, "n_derived": N2, "relation_targets": K}``。

    ``include_relations``/``relation_*`` 参数均从 :mod:`inference.config` 默认值取，
    显式传入可在 CLI 或测试中覆盖。
    """

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"知识根目录不存在: {root_dir}")

    include_relations = (
        config.INCLUDE_HIGHLIGHTED_RELATIONS_IN_INDEX
        if include_relations is None else bool(include_relations)
    )
    relation_max_depth = (
        config.HIGHLIGHT_INDEX_MAX_DEPTH
        if relation_max_depth is None else int(relation_max_depth)
    )
    relation_max_nodes = (
        config.HIGHLIGHT_INDEX_MAX_NODES
        if relation_max_nodes is None else int(relation_max_nodes)
    )
    relation_allow_remote = (
        config.HIGHLIGHT_INDEX_ALLOW_REMOTE
        if relation_allow_remote is None else bool(relation_allow_remote)
    )
    relation_remote_timeout = (
        config.HIGHLIGHT_INDEX_REMOTE_TIMEOUT
        if relation_remote_timeout is None else float(relation_remote_timeout)
    )

    original_chunks = build_knowledge_chunks(root_dir, chunk_size=chunk_size)
    n_original = len(original_chunks)
    if not original_chunks:
        logger.warning("[Indexer] %s 切块为空（原始 knowledge.md 无内容）", root_dir)

    derived_chunks: list[KnowledgeChunk] = []
    relation_targets: list[dict] = []
    if include_relations:
        try:
            derived_chunks, relation_targets = _collect_relation_chunks(
                root_dir,
                original_chunks,
                chunk_size=chunk_size,
                max_depth=relation_max_depth,
                max_nodes=relation_max_nodes,
                allow_remote=relation_allow_remote,
                remote_timeout=relation_remote_timeout,
            )
        except Exception as e:
            # 高亮关联属于增强能力，单个 root 抓取失败不应阻塞主索引构建。
            logger.warning(
                "[Indexer] 关联展开整体失败（保留原始 chunks 兜底）: %s", e,
                exc_info=True,
            )
            derived_chunks, relation_targets = [], []
    else:
        logger.info("[Indexer] INCLUDE_HIGHLIGHTED_RELATIONS_IN_INDEX=False，跳过关联展开")

    chunks: list[KnowledgeChunk] = list(original_chunks) + list(derived_chunks)
    n_derived = len(derived_chunks)

    chunks_path = os.path.join(root_dir, "_chunks.jsonl")
    bm25_path = os.path.join(root_dir, "_bm25.pkl")
    emb_path = os.path.join(root_dir, "_embeddings.npy")
    meta_path = os.path.join(root_dir, "_index_meta.json")
    targets_path = os.path.join(root_dir, "_relation_targets.json")

    _write_chunks_jsonl(chunks_path, chunks)
    logger.info(
        "[Indexer] 写入 %s（%d chunks：原始 %d + 派生 %d）",
        chunks_path, len(chunks), n_original, n_derived,
    )

    bm25_ok = False
    if chunks:
        try:
            bm25_index = bm25_mod.build([c.content for c in chunks])
            bm25_mod.save(bm25_index, bm25_path)
            bm25_ok = True
            logger.info("[Indexer] 写入 %s", bm25_path)
        except Exception as e:
            logger.warning("[Indexer] 构建 BM25 失败: %s", e)

    emb_ok = False
    if chunks and not skip_embedding:
        try:
            from ..embedding_client import EmbeddingNotConfigured, embed_texts

            try:
                vecs = await embed_texts(
                    [c.content for c in chunks],
                    model=embedding_model,
                    batch_size=embedding_batch_size,
                )
            except EmbeddingNotConfigured as e:
                logger.warning("[Indexer] 跳过 embedding（%s）", e)
                vecs = []
            if vecs:
                semantic_mod.save(vecs, emb_path)
                emb_ok = True
                logger.info("[Indexer] 写入 %s", emb_path)
        except Exception as e:
            logger.warning("[Indexer] 构建 embedding 失败: %s", e)

    # meta + targets：写在最后，确保只要 _index_meta.json 存在且 schema 对得上，
    # 三件套就一定是完整一致的。中途失败的话 meta 不会被写，stale 兜底会重建。
    _write_json_atomic(meta_path, {
        "schema_version": config.INDEX_SCHEMA_VERSION,
        "with_relations": bool(include_relations),
        "n_original": n_original,
        "n_derived": n_derived,
        "stale": False,
        "built_at": int(time.time() * 1000),
    })
    if include_relations:
        _write_json_atomic(targets_path, {
            "schema_version": config.INDEX_SCHEMA_VERSION,
            "targets": relation_targets,
        })

    return {
        "chunks": len(chunks),
        "bm25": bm25_ok,
        "embeddings": emb_ok,
        "n_original": n_original,
        "n_derived": n_derived,
        "relation_targets": len(relation_targets),
    }

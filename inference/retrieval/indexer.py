"""离线建索引（HTTP 版）：扫 ``page_knowledge/{root}/`` 切块 + 高亮外链派生 chunk +
客户端 jieba 分词 + 客户端 embedding，**整批 upsert 到 retrieval_service**。

入口 :func:`build_for_root`：

1. 调 ``knowledge_core/chunk_builder.build_knowledge_chunks(root, chunk_size)`` 切块；
2. 复用 ``knowledge_core/relation_crawler.RelationCrawler``（``expand_all=True``，
   纯静态展开、不走 LLM）扫该 root 下所有 ``clause.json`` 的外链 ``highlightedContent``
   引用图，多跳 BFS 抓回关联条款，渲染成派生 ``KnowledgeChunk`` 与原始 chunks 一并 upsert；
3. 客户端 :func:`inference.retrieval.bm25.tokenize_join` 计算 ``content_tokenized``；
4. 调 :func:`inference.embedding_client.embed_texts` 批量取向量；
5. 通过 :class:`inference.retrieval.client.RetrievalServiceClient` 一次性 ``upsert(mode="overwrite")``；
6. 落 ``page_knowledge/{root}/_index_meta__cs{chunk_size}.json``
   （schema_version + stale 标记 + built_at + chunk_size），供 ``app.py`` 的 cascade
   stale 标记机制按 ``(root, chunkSize)`` 维度精准失效。``_relation_targets.json``
   不再写——反向追踪通过 ``client.lookup_dependents`` 在线查询；同 root 下不同
   chunkSize 的索引落到独立 LanceDB 表（``{policy_id}__cs{chunk_size}``）互不覆盖。

幂等：``mode=overwrite`` 整表替换，不增量。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from knowledge_core.registries import RelationFragment, RelationRegistry
from knowledge_core.chunk_builder import (
    KnowledgeChunk,
    build_knowledge_chunks,
    split_relations_into_chunks,
)
from knowledge_core.clause_locator import ClauseLocator
from knowledge_core.relation_crawler import RelationCrawler

from utils.helpers import resolve_page_knowledge_dir as _resolve_page_knowledge_dir

from .. import config
from . import bm25 as bm25_mod
from .client import get_default_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- chunk_size 维度的索引身份
#
# 同一 policyId 下可能存在多套 chunkSize 的索引（``/api/inference/stream`` 允许
# 按请求覆盖）。为了让它们在 LanceDB 中互不覆盖，约定：
#
#   - 服务端表名 = ``{base_policy_id}__cs{chunk_size}``（``make_index_policy_id``）
#   - 元数据文件 = ``page_knowledge/<root>/_index_meta__cs{chunk_size}.json``
#     （``index_meta_filename``）
#
# ``parse_index_policy_id`` 用于 cascade 反查：lookup_dependents 返回的
# ``source_policy_id`` 是已带后缀的，需要拆回 ``(base_policy_id, chunk_size)``
# 才能定位到 page_knowledge 目录与对应的 meta 文件。


_INDEX_POLICY_ID_PATTERN = re.compile(r"^(.*)__cs(\d+)$")


def make_index_policy_id(policy_id: str, chunk_size: int) -> str:
    """把基础 ``policy_id`` 与 ``chunk_size`` 拍成 LanceDB 表名。"""

    return f"{policy_id}__cs{int(chunk_size)}"


def index_meta_filename(chunk_size: int) -> str:
    """``_index_meta__cs{chunk_size}.json``：同 root 下按 chunk_size 隔离的 meta 文件名。"""

    return f"_index_meta__cs{int(chunk_size)}.json"


def parse_index_policy_id(index_policy_id: str) -> tuple[str, Optional[int]]:
    """``A__cs500`` -> ``("A", 500)``；无后缀返回 ``(index_policy_id, None)``。"""

    m = _INDEX_POLICY_ID_PATTERN.match(index_policy_id or "")
    if not m:
        return index_policy_id, None
    return m.group(1), int(m.group(2))


# ---------------------------------------------------------------- policy_id 解析


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _page_knowledge_dir() -> str:
    """运行期解析 page_knowledge 根目录（读 ``PAGE_KNOWLEDGE_DIR`` 环境变量）。

    用函数而非模块级常量：``app.py`` 在 ``python app.py`` 形态下回写环境变量发生在本模块
    被延迟 import 之前/之后均可，运行期解析能稳妥拿到部署形态下的 ../resources/page_knowledge。
    """
    return _resolve_page_knowledge_dir(_PROJECT_ROOT)


def _policy_index_file() -> str:
    return os.path.join(_page_knowledge_dir(), "_policy_index.json")


def _write_index_meta(root_dir: str, payload: dict, *, chunk_size: int) -> None:
    """整文件原子写 ``_index_meta__cs{chunk_size}.json``，失败仅记 WARN
    （缺 meta 会被 stale 兜底重建）。"""

    meta_path = os.path.join(root_dir, index_meta_filename(chunk_size))
    try:
        os.makedirs(os.path.dirname(os.path.abspath(meta_path)), exist_ok=True)
        tmp = meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, meta_path)
    except OSError as e:
        logger.warning("[Indexer] 写 %s 失败（忽略）: %s", meta_path, e)


def _resolve_policy_id_for_root(root_dir: str) -> Optional[str]:
    """从 ``_policy_index.json`` 反查 ``root_dir`` 对应的 policy_id。

    ``app.py`` 调 ``build_for_root`` 时只给 root_dir，但服务端表是按 policy_id 命名的。
    本函数把 ``page_knowledge/{root_dirname}`` 映射回 policy_id；找不到时退化为 dirname
    本身（保留可回查的 trace），但更建议调用方显式传 ``policy_id``。
    """

    page_knowledge_dir = _page_knowledge_dir()
    abs_root = os.path.abspath(root_dir)
    if not abs_root.startswith(os.path.abspath(page_knowledge_dir)):
        return None
    rel = os.path.relpath(abs_root, page_knowledge_dir)
    rel = rel.split(os.sep)[0]  # 取顶层 dirname

    from extractor.policy_index import get_root_map

    root_map = get_root_map(_policy_index_file())  # {policy_id: dirname}
    for pid, dn in root_map.items():
        if dn == rel:
            return pid
    return rel  # 兜底：用 dirname 作伪 policy_id


def resolve_root_dir(policy_id: str) -> Optional[str]:
    """``policy_id`` -> ``page_knowledge/<dirname>`` 绝对路径；找不到返回 ``None``。

    供 :mod:`inference.scripts.build_indices` 等 CLI 使用，避免它们再去 import
    ``inference.retrieval.hybrid`` 模块级私有常量。
    """

    from extractor.policy_index import get_root_map

    root_map = get_root_map(_policy_index_file())
    dirname = root_map.get(policy_id)
    if not dirname:
        return None
    abs_dir = os.path.join(_page_knowledge_dir(), dirname)
    return abs_dir if os.path.isdir(abs_dir) else None


# ---------------------------------------------------------------- 关联展开


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

    复用 knowledge_core 的 RelationCrawler，但这里 ``expand_all=True`` +
    ``question=""``：跳过所有 LLM 判定，定位到内容即入
    fragment。本函数**不修改** original_chunks，但派生 chunk 的 ``parent_chunk_index``
    会指向最近的原始 chunk（按 source_dir 反查），便于 trace；定位不到时填 -1。

    返回 (derived_chunks, relation_targets)：

    - derived_chunks：已分配最终 index（从 ``len(original_chunks)+1`` 起递增），
      可直接 ``extend`` 进 chunks 列表后 upsert。
    - relation_targets：``[{"policy_id": ..., "clause_id": ...}, ...]`` 的反向追踪表，
      仅作为日志返回——cascade 触发器现在直接走 ``client.lookup_dependents``。
    """
    page_knowledge_dir = os.path.dirname(os.path.abspath(root_dir))
    policy_index_path = os.path.join(page_knowledge_dir, "_policy_index.json")

    locator = ClauseLocator(
        page_knowledge_dir=page_knowledge_dir,
        policy_index_path=policy_index_path,
        remote_timeout=remote_timeout,
    )
    if not allow_remote:
        locator._try_remote = lambda _pid, _cid: None  # type: ignore[assignment]

    registry = RelationRegistry()
    dir_to_chunk_index: dict[str, int] = {}
    for c in original_chunks:
        for d in c.directories or []:
            dir_to_chunk_index.setdefault(os.path.abspath(d), c.index)

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

    by_parent: dict[int, list[RelationFragment]] = {}
    for frag in all_fragments:
        by_parent.setdefault(frag.parent_chunk_index, []).append(frag)

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
        for dc in derived:
            dc.index = next_index
            next_index += 1
            derived_chunks.append(dc)
        derived_seq_per_parent[parent_idx] = start_seq + len(derived) - 1

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


# ---------------------------------------------------------------- embedding


async def _embed_texts_safe(
    texts: list[str],
    *,
    batch_size: int,
    model: Optional[str],
) -> tuple[list[list[float]], int]:
    """安全 embed：未配置 / 失败时返回 ``([], 0)``，由调用方降级到纯 BM25。"""

    if not texts:
        return [], 0
    try:
        from ..embedding_client import EmbeddingNotConfigured, embed_texts
    except Exception as e:  # pragma: no cover
        logger.warning("[Indexer] 导入 embedding_client 失败: %s", e)
        return [], 0
    try:
        vecs = await embed_texts(texts, model=model, batch_size=batch_size)
    except EmbeddingNotConfigured as e:
        logger.warning("[Indexer] 跳过 embedding（%s）", e)
        return [], 0
    except Exception as e:
        logger.warning("[Indexer] embed_texts 失败: %s", e)
        return [], 0
    if not vecs:
        return [], 0
    dim = len(vecs[0]) if vecs[0] else 0
    return [list(v) for v in vecs], dim


# ---------------------------------------------------------------- 主入口


async def build_for_root(
    root_dir: str,
    *,
    policy_id: str | None = None,
    chunk_size: int = config.INFERENCE_DEFAULT_CHUNK_SIZE,
    embedding_batch_size: int = 10,
    embedding_model: Optional[str] = None,
    skip_embedding: bool = False,
    include_relations: Optional[bool] = None,
    relation_max_depth: Optional[int] = None,
    relation_max_nodes: Optional[int] = None,
    relation_allow_remote: Optional[bool] = None,
    relation_remote_timeout: Optional[float] = None,
) -> dict:
    """对单个知识根目录建全套索引，整表 overwrite 到 retrieval_service。

    ``policy_id`` 为业务侧基础 ID（与 ``_policy_index.json`` 对齐）；本函数内部会
    根据 ``chunk_size`` 派生服务端表名 ``make_index_policy_id(policy_id, chunk_size)``，
    并把 meta 文件落到 ``_index_meta__cs{chunk_size}.json``。同一 root 下多套
    chunkSize 的索引互不覆盖。

    返回 ``{"chunks", "n_original", "n_derived", "relation_targets",
    "embeddings": bool, "policy_id"（基础 ID）, "index_policy_id"（服务端表名）,
    "chunk_size"}``。
    """

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"知识根目录不存在: {root_dir}")

    base_pid = policy_id or _resolve_policy_id_for_root(root_dir)
    if not base_pid:
        raise ValueError(f"无法解析 policy_id: root_dir={root_dir}")
    pid = make_index_policy_id(base_pid, chunk_size)

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

    # 1) 切块
    original_chunks: list[KnowledgeChunk] = await asyncio.to_thread(
        build_knowledge_chunks, root_dir, chunk_size
    )
    n_original = len(original_chunks)
    if not original_chunks:
        logger.warning("[Indexer] %s 切块为空（原始 knowledge.md 无内容）", root_dir)

    # 2) 高亮外链派生
    derived_chunks: list[KnowledgeChunk] = []
    relation_targets: list[dict] = []
    if include_relations:
        try:
            derived_chunks, relation_targets = await asyncio.to_thread(
                _collect_relation_chunks,
                root_dir,
                original_chunks,
                chunk_size=chunk_size,
                max_depth=relation_max_depth,
                max_nodes=relation_max_nodes,
                allow_remote=relation_allow_remote,
                remote_timeout=relation_remote_timeout,
            )
        except Exception as e:
            logger.warning(
                "[Indexer] 关联展开整体失败（保留原始 chunks 兜底）: %s", e,
                exc_info=True,
            )
            derived_chunks, relation_targets = [], []
    else:
        logger.info("[Indexer] INCLUDE_HIGHLIGHTED_RELATIONS_IN_INDEX=False，跳过关联展开")

    chunks: list[KnowledgeChunk] = list(original_chunks) + list(derived_chunks)
    n_derived = len(derived_chunks)

    if not chunks:
        client = await get_default_client()
        try:
            await client.drop_policy(pid)
        except Exception as e:
            logger.debug("[Indexer] drop 空 policy 失败（忽略）: %s", e)
        _write_index_meta(root_dir, {
            "schema_version": config.INDEX_SCHEMA_VERSION,
            "policy_id": base_pid,
            "index_policy_id": pid,
            "chunk_size": int(chunk_size),
            "with_relations": bool(include_relations),
            "n_original": 0,
            "n_derived": 0,
            "embeddings": False,
            "stale": False,
            "built_at": int(time.time() * 1000),
        }, chunk_size=chunk_size)
        return {
            "policy_id": base_pid,
            "index_policy_id": pid,
            "chunk_size": int(chunk_size),
            "chunks": 0,
            "n_original": 0,
            "n_derived": 0,
            "relation_targets": 0,
            "embeddings": False,
        }

    # 3) 客户端分词
    tokenized: list[str] = await asyncio.to_thread(
        lambda: [bm25_mod.tokenize_join(c.content or "") for c in chunks]
    )

    # 4) 客户端 embedding
    vectors: list[list[float]] = []
    dim = 0
    if not skip_embedding:
        vectors, dim = await _embed_texts_safe(
            [c.content or "" for c in chunks],
            batch_size=embedding_batch_size,
            model=embedding_model,
        )

    emb_ok = bool(vectors)

    # 5) HTTP upsert
    client = await get_default_client()
    result = await client.upsert_knowledge_chunks(
        pid,
        chunks,
        tokenized=tokenized,
        vectors=vectors,
        mode="overwrite",
        expected_dim=dim if emb_ok else None,
    )
    logger.info(
        "[Indexer] policy=%s upsert 完成: %s（n_original=%d n_derived=%d emb=%s）",
        pid, result, n_original, n_derived, emb_ok,
    )

    # meta 在最后一步写：upsert 成功后才标记"非 stale"。中途失败 → meta 不更新，
    # _inference_artifacts_stale 兜底重建。
    _write_index_meta(root_dir, {
        "schema_version": config.INDEX_SCHEMA_VERSION,
        "policy_id": base_pid,
        "index_policy_id": pid,
        "chunk_size": int(chunk_size),
        "with_relations": bool(include_relations),
        "n_original": n_original,
        "n_derived": n_derived,
        "embeddings": emb_ok,
        "stale": False,
        "built_at": int(time.time() * 1000),
    }, chunk_size=chunk_size)

    return {
        "policy_id": base_pid,
        "index_policy_id": pid,
        "chunk_size": int(chunk_size),
        "chunks": int(result.get("table_size", len(chunks))),
        "n_original": n_original,
        "n_derived": n_derived,
        "relation_targets": len(relation_targets),
        "embeddings": emb_ok,
    }


# ---------------------------------------------------------------- 差量 embedding 自愈


async def rebuild_embeddings_only(
    policy_id: str,
    *,
    embedding_batch_size: int = 10,
    embedding_model: Optional[str] = None,
) -> dict:
    """差量补建：拉服务端现有 chunks（不动 content/tokenized），重算向量后 merge 回写。

    适用场景：``build_for_root`` 整体跑成功但 embedding 步骤失败（embedding 服务暂时
    异常 / 网络抖动），后续 ``hybrid_search`` 发现服务端 ``meta.dim=0`` 时
    fire-and-forget 触发本函数自愈。

    返回 ``{"embeddings": bool, "n": int}``：``embeddings`` 表示是否成功写盘，
    ``n`` 是本次输入的 chunk 数。
    """

    client = await get_default_client()
    rows = await client.list_chunks(policy_id, limit=100000, include_content=True)
    if not rows:
        return {"embeddings": False, "n": 0, "error": "no chunks"}

    contents = [r.get("content") or "" for r in rows]
    vectors, dim = await _embed_texts_safe(
        contents,
        batch_size=embedding_batch_size,
        model=embedding_model,
    )
    if not vectors:
        return {"embeddings": False, "n": len(rows), "error": "empty vectors"}

    # 直接走低级 upsert（merge_by_chunk_id）：仅更新 vector 列；其余字段也带上避免缺列。
    payload: list[dict] = []
    for r, v in zip(rows, vectors):
        rks = r.get("relation_keys") or []
        payload.append({
            "chunk_id": int(r["chunk_id"]),
            "content": r.get("content") or "",
            "content_tokenized": "",  # 服务端 merge 不会清空——我们这里写空就会清。
            "vector": list(v),
            "heading_paths": list(r.get("heading_paths") or []),
            "directories": list(r.get("directories") or []),
            "kind": r.get("kind") or "original",
            "parent_chunk_index": int(r.get("parent_chunk_index", -1)),
            "derived_seq": int(r.get("derived_seq", 0)),
            "relation_keys": [
                {"policy_id": rk.get("policy_id", ""), "clause_id": rk.get("clause_id", "")}
                for rk in rks
            ],
            "hop_depth": int(r.get("hop_depth", 0)),
            "source": r.get("source") or "",
            "clause_id": r.get("clause_id") or "",
            "built_at": 0,
        })

    # NOTE：``merge_by_chunk_id`` 会用 payload 整行替换匹配 chunk_id 的现有行，所以
    # 这里必须把 ``content_tokenized`` 也一起带上。差量补 vector 时，重新算
    # tokenized 不会改变结果（同一个 content + 同一份 tokenize 函数），代价是 CPU 而已。
    from . import bm25 as bm25_mod
    for row in payload:
        row["content_tokenized"] = bm25_mod.tokenize_join(row["content"])

    result = await client.upsert_chunks(
        policy_id,
        payload,
        mode="merge_by_chunk_id",
        expected_dim=dim,
    )
    return {
        "embeddings": True,
        "n": len(rows),
        "table_size": result.get("table_size", 0),
        "dim": dim,
    }

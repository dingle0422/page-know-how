"""离线建索引：扫 ``page_knowledge/{root}/`` 切块 + BM25 + embedding 落盘。

入口 :func:`build_for_root`：

1. 调 ``reasoner/v3/chunk_builder.build_knowledge_chunks(root, chunk_size)`` 切块；
2. 写 ``_chunks.jsonl``；
3. 用 :mod:`inference.retrieval.bm25` 建 BM25 并写 ``_bm25.pkl``；
4. 调 :func:`inference.embedding_client.embed_texts` 批量取向量并写 ``_embeddings.npy``，
   embedding 服务未配置时跳过该步骤（hybrid 检索会自动退化为纯 BM25）。

幂等：每次都整文件覆盖，不增量。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from reasoner.v3.chunk_builder import KnowledgeChunk, build_knowledge_chunks

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


async def build_for_root(
    root_dir: str,
    *,
    chunk_size: int = config.CHUNK_SIZE,
    # 服务端单次最多接收 10 条；这里默认就贴着上限，避免 embedding_client 内
    # 二次夹断时再打一条 INFO。如有更小限制可手工传入更小值。
    embedding_batch_size: int = 10,
    embedding_model: Optional[str] = None,
    skip_embedding: bool = False,
) -> dict:
    """对单个知识根目录建全套索引，返回 ``{"chunks": N, "bm25": bool, "embeddings": bool}``。"""

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"知识根目录不存在: {root_dir}")

    chunks = build_knowledge_chunks(root_dir, chunk_size=chunk_size)
    if not chunks:
        logger.warning("[Indexer] %s 切块为空，仅写入空 chunks 文件", root_dir)

    chunks_path = os.path.join(root_dir, "_chunks.jsonl")
    bm25_path = os.path.join(root_dir, "_bm25.pkl")
    emb_path = os.path.join(root_dir, "_embeddings.npy")

    _write_chunks_jsonl(chunks_path, chunks)
    logger.info("[Indexer] 写入 %s（%d chunks）", chunks_path, len(chunks))

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

    return {"chunks": len(chunks), "bm25": bm25_ok, "embeddings": emb_ok}

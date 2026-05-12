"""离线建索引 CLI。

用法::

    # 按 policyId 建索引（自动从 page_knowledge/_policy_index.json 解析 root）
    python -m inference.scripts.build_indices --policy-id <id>

    # 直接指定知识根目录（page_knowledge/<dirname>）
    python -m inference.scripts.build_indices --root <abs_or_relative_dir>

    # 跳过 embedding 步骤（仅建 BM25）
    python -m inference.scripts.build_indices --policy-id <id> --skip-embedding

环境变量：

- ``INFERENCE_EMBEDDING_URL`` / ``INFERENCE_EMBEDDING_MODEL`` /
  ``INFERENCE_EMBEDDING_AUTH``：embedding 服务配置。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from .. import config
from ..retrieval.indexer import (
    _PAGE_KNOWLEDGE_DIR,
    build_for_root,
    resolve_root_dir as _resolve_root_dir,
)


def _resolve_root(args: argparse.Namespace) -> str:
    if args.root:
        return args.root if os.path.isabs(args.root) else os.path.abspath(args.root)
    if not args.policy_id:
        raise SystemExit("必须指定 --policy-id 或 --root")
    root = _resolve_root_dir(args.policy_id)
    if not root:
        raise SystemExit(
            f"policy_id={args.policy_id} 未在 _policy_index.json 中找到 root，"
            f"请先调 /api/extract 抽取，或显式传 --root"
        )
    return root


async def _amain(args: argparse.Namespace) -> int:
    root = _resolve_root(args)
    include_relations = not args.no_relations
    allow_remote = not args.no_remote
    logging.info(
        "[BuildIndices] root=%s chunk_size=%d skip_embedding=%s "
        "include_relations=%s max_depth=%d max_nodes=%d allow_remote=%s "
        "page_knowledge=%s",
        root, args.chunk_size, args.skip_embedding,
        include_relations, args.relation_max_depth, args.relation_max_nodes, allow_remote,
        _PAGE_KNOWLEDGE_DIR,
    )
    result = await build_for_root(
        root,
        chunk_size=args.chunk_size,
        embedding_batch_size=args.batch_size,
        embedding_model=args.embedding_model,
        skip_embedding=args.skip_embedding,
        include_relations=include_relations,
        relation_max_depth=args.relation_max_depth,
        relation_max_nodes=args.relation_max_nodes,
        relation_allow_remote=allow_remote,
        relation_remote_timeout=args.relation_remote_timeout,
    )
    print(
        f"[done] policy={result['policy_id']} chunks={result['chunks']} "
        f"(original={result['n_original']} derived={result['n_derived']}) "
        f"embeddings={'ok' if result['embeddings'] else 'skip'} "
        f"relation_targets={result['relation_targets']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="为单个 policy 构建 BM25 + 向量索引")
    parser.add_argument("--policy-id", help="policyId（与 --root 二选一）")
    parser.add_argument("--root", help="知识根目录绝对/相对路径（与 --policy-id 二选一）")
    parser.add_argument(
        "--chunk-size", type=int, default=config.CHUNK_SIZE,
        help=f"切块字符上限（默认 {config.CHUNK_SIZE}）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="embedding 调用批大小（默认 10，服务端硬上限就是 10，"
             "传更大会被 embedding_client 强行夹到 10）",
    )
    parser.add_argument(
        "--embedding-model", default=None,
        help="覆盖 INFERENCE_EMBEDDING_MODEL 环境变量",
    )
    parser.add_argument(
        "--skip-embedding", action="store_true",
        help="跳过 embedding 步骤（仅建 BM25）",
    )
    parser.add_argument(
        "--no-relations", action="store_true",
        help="不烘入高亮外链派生 chunk（退化为旧行为，便于回归对照）",
    )
    parser.add_argument(
        "--relation-max-depth", type=int,
        default=config.HIGHLIGHT_INDEX_MAX_DEPTH,
        help=f"关联展开 BFS 最大深度（默认 {config.HIGHLIGHT_INDEX_MAX_DEPTH}）",
    )
    parser.add_argument(
        "--relation-max-nodes", type=int,
        default=config.HIGHLIGHT_INDEX_MAX_NODES,
        help=f"单次 crawl 节点预算（默认 {config.HIGHLIGHT_INDEX_MAX_NODES}）",
    )
    parser.add_argument(
        "--no-remote", action="store_true",
        help="禁用远程兜底：本地未命中的 (policy_id, clause_id) 直接跳过",
    )
    parser.add_argument(
        "--relation-remote-timeout", type=float,
        default=config.HIGHLIGHT_INDEX_REMOTE_TIMEOUT,
        help=f"远程兜底单条超时秒（默认 {config.HIGHLIGHT_INDEX_REMOTE_TIMEOUT}）",
    )
    args = parser.parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())

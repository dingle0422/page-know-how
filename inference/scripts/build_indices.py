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
from ..retrieval.hybrid import _PAGE_KNOWLEDGE_DIR, _resolve_root_dir
from ..retrieval.indexer import build_for_root


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
    logging.info(
        "[BuildIndices] root=%s chunk_size=%d skip_embedding=%s page_knowledge=%s",
        root, args.chunk_size, args.skip_embedding, _PAGE_KNOWLEDGE_DIR,
    )
    result = await build_for_root(
        root,
        chunk_size=args.chunk_size,
        embedding_batch_size=args.batch_size,
        embedding_model=args.embedding_model,
        skip_embedding=args.skip_embedding,
    )
    print(
        f"[done] chunks={result['chunks']} "
        f"bm25={'ok' if result['bm25'] else 'skip'} "
        f"embeddings={'ok' if result['embeddings'] else 'skip'}"
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
        "--batch-size", type=int, default=32,
        help="embedding 调用批大小（默认 32）",
    )
    parser.add_argument(
        "--embedding-model", default=None,
        help="覆盖 INFERENCE_EMBEDDING_MODEL 环境变量",
    )
    parser.add_argument(
        "--skip-embedding", action="store_true",
        help="跳过 embedding 步骤（仅建 BM25）",
    )
    args = parser.parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())

"""检索子模块（HTTP 化版）。

存储 + 检索均迁移到独立的 ``retrieval_service`` 微服务（基于 LanceDB）。
本包只剩"客户端壳"职责：

- :mod:`.bm25`：jieba 分词（与服务端 FTS 完全同源）；
- :mod:`.client`：``RetrievalServiceClient`` HTTP SDK；
- :mod:`.hybrid`：对外暴露 ``hybrid_search``；
- :mod:`.indexer`：``build_for_root`` / ``rebuild_embeddings_only``。

服务端见仓库根目录 ``retrieval_service/``。
"""

from .hybrid import hybrid_search, invalidate

__all__ = ["hybrid_search", "invalidate"]

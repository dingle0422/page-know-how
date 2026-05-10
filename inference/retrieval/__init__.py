"""检索子模块。

对外暴露 :func:`hybrid_search`，内部分别封装 BM25 / 语义 / RRF。

约定的索引落盘位置（以 policy 的知识根目录为锚点）：

- ``page_knowledge/{root}/_chunks.jsonl``：每行一个 chunk 元数据
- ``page_knowledge/{root}/_bm25.pkl``：tokenized corpus + BM25Okapi 实例
- ``page_knowledge/{root}/_embeddings.npy``：``(N, dim)`` float32 向量库

未生成索引时 :func:`hybrid_search` 自动回退到纯 BM25 + 警告日志，不阻塞流程。
"""

from .hybrid import hybrid_search

__all__ = ["hybrid_search"]

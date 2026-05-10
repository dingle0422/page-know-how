"""语义检索：加载 ``_embeddings.npy``，用查询向量与库做余弦相似度。

落盘文件：``page_knowledge/{root}/_embeddings.npy``，shape ``(N, dim) float32``，
顺序与 ``_chunks.jsonl`` 严格对齐。

异步入口 :func:`search` 会内部调 :func:`inference.embedding_client.embed_texts`
拿到 query embedding，然后调本模块的纯计算函数 :func:`search_with_query_vec`。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def load(path: str):
    """加载 ``.npy``，返回 ndarray 或 None。"""

    if not os.path.exists(path):
        return None
    try:
        import numpy as np  # type: ignore
    except ImportError:
        logger.warning("[Semantic] 缺少 numpy，无法加载向量库")
        return None
    try:
        arr = np.load(path)
    except Exception as e:
        logger.warning("[Semantic] 加载向量库失败 %s: %s", path, e)
        return None
    if arr.ndim != 2:
        logger.warning("[Semantic] 向量库 shape 非法 %s", getattr(arr, "shape", None))
        return None
    return arr.astype("float32", copy=False)


def save(matrix, path: str) -> None:
    import numpy as np  # type: ignore

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = np.asarray(matrix, dtype="float32")
    np.save(path, arr)


def search_with_query_vec(
    matrix,
    query_vec: list[float],
    top_k: int,
) -> list[tuple[int, float]]:
    """纯计算：返回 ``[(chunk_idx, cosine_score), ...]``。"""

    if matrix is None:
        return []
    import numpy as np  # type: ignore

    q = np.asarray(query_vec, dtype="float32")
    if q.ndim != 1 or q.shape[0] != matrix.shape[1]:
        logger.warning(
            "[Semantic] query 维度不匹配 q=%s matrix=%s", q.shape, matrix.shape
        )
        return []
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return []
    q = q / q_norm
    matrix_norm = np.linalg.norm(matrix, axis=1)
    matrix_norm[matrix_norm == 0.0] = 1.0
    sims = (matrix @ q) / matrix_norm
    if top_k <= 0:
        return []
    idx = np.argpartition(-sims, kth=min(top_k, sims.shape[0] - 1))[:top_k]
    pairs = [(int(i), float(sims[i])) for i in idx]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


async def search(
    matrix,
    query: str,
    top_k: int,
    *,
    model: Optional[str] = None,
) -> list[tuple[int, float]]:
    """异步：内部走 embedding 服务取 query 向量后做检索。

    embedding 服务未配置时直接返回 ``[]``，由上层 hybrid 层降级。
    """

    if matrix is None or not query:
        return []
    from ..embedding_client import EmbeddingNotConfigured, embed_texts

    try:
        vecs = await embed_texts([query], model=model)
    except EmbeddingNotConfigured as e:
        logger.warning("[Semantic] embedding 未配置，跳过语义检索: %s", e)
        return []
    except Exception as e:
        logger.warning("[Semantic] 取 query 向量失败: %s", e)
        return []
    if not vecs:
        return []
    return search_with_query_vec(matrix, vecs[0], top_k)

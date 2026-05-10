"""BM25 索引：jieba 分词 + rank_bm25.BM25Okapi。

落盘文件：``page_knowledge/{root}/_bm25.pkl``，内部为 dict::

    {
        "tokenized_corpus": list[list[str]],
        "bm25": BM25Okapi 实例,
    }

下游通过 :func:`load` 懒加载、:func:`tokenize` 复用同一套分词。
"""

from __future__ import annotations

import logging
import os
import pickle
import re
from typing import Any

logger = logging.getLogger(__name__)

# 极小化的中文停用词集合，避免引入额外文件；后续若需要扩展可换成从 .txt 读。
_STOPWORDS = {
    "的", "了", "和", "是", "在", "也", "或", "与", "及", "对",
    "为", "等", "其", "所", "此", "之", "上", "下", "中", "前后",
    "可以", "不能", "应当", "需要", "我们", "你们", "他们",
}

_NON_WORD = re.compile(r"[\s\u3000\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """统一分词入口：建索引和查询都走这里，确保 token 集对齐。"""

    if not text:
        return []
    # 延迟导入：缺 jieba 时给出明确错误，避免 import inference.retrieval 直接挂掉。
    import jieba  # type: ignore

    raw = jieba.lcut(text)
    out: list[str] = []
    for tok in raw:
        tok = tok.strip().lower()
        if not tok:
            continue
        if _NON_WORD.fullmatch(tok):
            continue
        if tok in _STOPWORDS:
            continue
        out.append(tok)
    return out


def build(corpus_texts: list[str]) -> dict[str, Any]:
    """根据原始 chunk 文本列表构建 BM25 索引（不落盘）。"""

    from rank_bm25 import BM25Okapi  # type: ignore

    tokenized = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    return {"tokenized_corpus": tokenized, "bm25": bm25}


def save(index: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        logger.warning("[BM25] 加载索引失败 %s: %s", path, e)
        return None
    if not isinstance(data, dict) or "bm25" not in data:
        logger.warning("[BM25] 索引文件结构异常: %s", path)
        return None
    return data


def search(index: dict[str, Any], query: str, top_k: int) -> list[tuple[int, float]]:
    """返回 ``[(chunk_idx, score), ...]``，按分数降序，取前 ``top_k``。"""

    if not query or not index:
        return []
    bm25 = index.get("bm25")
    if bm25 is None:
        return []
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    scores = bm25.get_scores(query_tokens)
    pairs = [(i, float(s)) for i, s in enumerate(scores)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[: max(int(top_k), 0)]

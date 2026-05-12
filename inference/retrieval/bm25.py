"""中文分词工具：jieba 分词 + 极小化停用词。

历史上这里还封装了 BM25 索引的 build/save/load/search；改造为 retrieval_service 后，
存储与检索都迁移到服务端 LanceDB。本模块只剩**纯客户端分词**职责，被以下三处共用，
确保写入与查询使用完全一致的分词逻辑：

- :func:`inference.retrieval.indexer.build_for_root`：构建 ``content_tokenized`` 列；
- :func:`inference.retrieval.hybrid.hybrid_search`：构建 ``query_tokenized`` 入参；
- :mod:`inference.retrieval.client`：序列化时调 :func:`tokenize_join`。

服务端 (``retrieval_service``) 不依赖 jieba，FTS 走 ``base_tokenizer="whitespace"`` 直接
吃这里产出的空格分隔字符串，分词质量 100% 与本模块同源。
"""

from __future__ import annotations

import re

# 极小化的中文停用词集合，避免引入额外文件；后续若需要扩展可换成从 .txt 读。
_STOPWORDS = {
    "的", "了", "和", "是", "在", "也", "或", "与", "及", "对",
    "为", "等", "其", "所", "此", "之", "上", "下", "中", "前后",
    "可以", "不能", "应当", "需要", "我们", "你们", "他们",
}

_NON_WORD = re.compile(r"[\s\u3000\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """统一分词入口：jieba.lcut + lower + 停用词/非词过滤。"""

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


def tokenize_join(text: str) -> str:
    """``" ".join(tokenize(text))``：服务端 FTS whitespace tokenizer 的喂料格式。"""

    return " ".join(tokenize(text))

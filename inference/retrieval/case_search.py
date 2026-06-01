"""preview 阶段 case 库纯向量检索。

供 :mod:`inference.preview` 在生成 preview prompt 之前调用，从 LanceDB v2
集合 ``case_{khCode}`` 拉历史经验，按 cosine_similarity 阈值过滤后取 top-k,
注入到 preview user prompt 的【历史经验】段。

集合命名约定（与 :mod:`case_refinery` 服务对齐）：

- ``khCode = policyId.split("_")[0]``
- ``collection_id = f"case_{khCode}"``
- inference pipeline 传入的 policy_id 已经被 ``make_index_policy_id`` 拼上
  ``__cs{N}`` 后缀，本模块内部会先剥后缀再 split。

检索协议（详见 :func:`inference.retrieval.client.RetrievalServiceClient.vector_search_v2`）：

- 走 ``POST /v2/collections/{cid}/search``，``query_tokenized=""`` + ``top_m=0``
  纯向量召回；响应 ``hits[*].cosine_similarity`` 直接给出原始 cosine 分。

极性分桶检索（positive / negative 各自独立召回）：

- 案例库里 positive 案例占绝对多数，单次混合检索的 top 命中几乎全是 positive,
  negative 经验会被淹没。因此对 ``case_polarity`` 做**分桶独立检索**：
  positive 召回 top-k、negative 召回 top-k，各自按阈值过滤后合并注入,
  保证两类经验都有代表。``top_k``（即请求体 ``topC``）是**每个极性**的配额,
  最终注入最多 ``2 * top_k`` 条。
- where 过滤走服务端扁平化列名 ``md_case_polarity_<hash8>``（**不能**用原始字段名
  ``case_polarity``，那样不报错但永远筛不到），列名从
  :meth:`RetrievalServiceClient.get_collection_meta_v2` 的 ``filterable_fields``
  动态解析并缓存；同时附加 ``md_tombstoned_<hash8> = false`` 排除已删除案例。
- 两层注入决策：①先 cosine ≥ ``threshold`` 过滤；②再按 ``top_k`` 截断。
  不足 / 为空都允许（preview 自动回退到不带【历史经验】段的 prompt）。

降级策略（全链路兜底）：

- policy_id 解析不出 kh_code → ``[]``；
- embedding 服务异常 → ``[]``；
- LanceDB 集合不存在（404）→ ``[]``（case_refinery 未上线是预期）；
- 服务不可达 / 5xx → ``[]``；
- meta 解析不出 polarity 列 → 退化为单次混合检索（不分桶）；
- 命中 0 条 ≥ 阈值 → ``[]``。

返回 ``[]`` 时 :func:`inference.prompts.select_preview_prompt` 会自动回退到
原 2 套 PREVIEW_* prompt（不渲染【历史经验】段），与 ``topC=0`` 体验一致。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "CaseHit",
    "search_cases",
    "resolve_case_collection_id",
]

# 扁平化列名前缀（服务端把 metadata.<field> 落成 md_<field>_<hash8> 列）。
_POLARITY_COL_PREFIX = "md_case_polarity_"
_TOMBSTONE_COL_PREFIX = "md_tombstoned_"

# 已知稳定列名（hash 基于路径，跨集合一致）。meta 拉取失败时作为兜底，
# 避免一次 meta 抖动就整体退化为不分桶检索。
_POLARITY_COL_FALLBACK = "md_case_polarity_af1bc6c9"
_TOMBSTONE_COL_FALLBACK = "md_tombstoned_45c36238"

_POLARITIES = ("positive", "negative")

# collection_id -> (polarity_col, tombstone_col)；列名稳定，进程内缓存即可。
_COLUMN_CACHE: dict[str, tuple[Optional[str], Optional[str]]] = {}


@dataclass
class CaseHit:
    """单条 case 召回结果（preview prompt 渲染所需字段集合）。"""

    cosine_similarity: float
    question: str
    knowledge: str
    polarity: str  # "positive" / "negative" / ""


def resolve_case_collection_id(policy_id: Optional[str]) -> str:
    """``policy_id`` → ``case_{khCode}`` 的标准映射。

    - 先剥 ``__cs{N}`` 后缀（inference 内部走的是 ``index_policy_id``）；
    - 再 ``split("_")[0]`` 取 khCode；
    - 任意一步拿不到 → 返回空串，由调用方据此跳过检索。
    """

    base = (policy_id or "").split("__cs")[0]
    kh_code = base.split("_")[0] if base else ""
    if not kh_code:
        return ""
    return f"case_{kh_code}"


def _coerce_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_knowledge(metadata: dict) -> str:
    """从 metadata 拼出案例知识正文。

    - 主字段 ``refined_knowledge`` 非空 → 直接用；
    - 为空（``refine_status=raw_fallback``）→ 退化为 ``answer_content`` +
      ``thinking`` 的拼接，保证总能给模型看到点东西；
    - 都为空 → 返回空串，调用方会跳过该条。
    """

    refined = _coerce_str(metadata.get("refined_knowledge"))
    if refined:
        return refined
    answer = _coerce_str(metadata.get("answer_content"))
    thinking = _coerce_str(metadata.get("thinking"))
    parts: list[str] = []
    if answer:
        parts.append(answer)
    if thinking:
        parts.append(f"思考过程：{thinking}")
    return "\n\n".join(parts)


def _hit_to_case(hit: dict) -> Optional[CaseHit]:
    """单条 hit → CaseHit；字段缺失或知识为空返回 None。"""

    sim_raw = hit.get("cosine_similarity")
    if sim_raw is None:
        return None
    try:
        sim = float(sim_raw)
    except (TypeError, ValueError):
        return None
    metadata = hit.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    knowledge = _extract_knowledge(metadata)
    if not knowledge:
        return None
    question = _coerce_str(metadata.get("question_content")) or _coerce_str(hit.get("content"))
    polarity_raw = _coerce_str(metadata.get("case_polarity")).lower()
    polarity = polarity_raw if polarity_raw in {"positive", "negative"} else ""
    return CaseHit(
        cosine_similarity=sim,
        question=question,
        knowledge=knowledge,
        polarity=polarity,
    )


async def _embed_question(question: str) -> list[float]:
    """复用 :mod:`inference.embedding_client`，失败 → ``[]``。"""

    try:
        from ..embedding_client import EmbeddingNotConfigured, embed_texts
    except Exception as e:  # pragma: no cover
        logger.warning("[CaseSearch] 导入 embedding_client 失败: %s", e)
        return []
    try:
        vecs = await embed_texts([question])
    except EmbeddingNotConfigured as e:
        logger.info("[CaseSearch] embedding 未配置，跳过 case 检索: %s", e)
        return []
    except Exception as e:
        logger.warning("[CaseSearch] embedding 失败，跳过 case 检索: %s", e)
        return []
    if not vecs:
        return []
    return list(vecs[0] or [])


async def _resolve_filter_columns(
    client, collection_id: str
) -> tuple[Optional[str], Optional[str]]:
    """解析 ``(polarity_col, tombstone_col)`` 扁平化列名，进程内缓存。

    - 先查缓存；
    - 拉 ``/v2/collections/{cid}/meta`` 的 ``filterable_fields``，按前缀匹配；
    - meta 失败 / 找不到列 → 回退到稳定的兜底列名（hash 基于路径，跨集合一致），
      保证不会因 meta 抖动就整体退化为不分桶检索。

    polarity_col 为 ``None`` 时调用方退化为单次混合检索。
    """

    cached = _COLUMN_CACHE.get(collection_id)
    if cached is not None:
        return cached

    polarity_col: Optional[str] = None
    tombstone_col: Optional[str] = None
    try:
        meta = await client.get_collection_meta_v2(collection_id)
    except Exception as e:
        logger.warning(
            "[CaseSearch] get_collection_meta_v2 失败 collection=%s: %s",
            collection_id, e,
        )
        meta = None

    if isinstance(meta, dict):
        fields = meta.get("filterable_fields") or []
        for f in fields:
            if not isinstance(f, str):
                continue
            if polarity_col is None and f.startswith(_POLARITY_COL_PREFIX):
                polarity_col = f
            elif tombstone_col is None and f.startswith(_TOMBSTONE_COL_PREFIX):
                tombstone_col = f

    # meta 拿不到列时用稳定兜底，让分桶检索仍能工作。
    if polarity_col is None:
        polarity_col = _POLARITY_COL_FALLBACK
    if tombstone_col is None:
        tombstone_col = _TOMBSTONE_COL_FALLBACK

    resolved = (polarity_col, tombstone_col)
    _COLUMN_CACHE[collection_id] = resolved
    return resolved


def _build_where(polarity: str, polarity_col: str, tombstone_col: Optional[str]) -> str:
    """拼极性 + 墓碑排除的 where 表达式。"""

    where = f"{polarity_col} = '{polarity}'"
    if tombstone_col:
        where += f" AND {tombstone_col} = false"
    return where


async def _search_one_bucket(
    client,
    collection_id: str,
    q_vec: list[float],
    *,
    where: Optional[str],
    threshold: float,
    top_k: int,
) -> list[CaseHit]:
    """单桶向量检索 + 阈值过滤 + 降序截 top_k。失败返回 ``[]``。"""

    # 多召回一些再阈值过滤，避免阈值卡得严时凑不齐 top_k。
    fetch_n = max(top_k * 3, top_k + 5)
    try:
        hits = await client.vector_search_v2(
            collection_id,
            query_vector=q_vec,
            top_n=fetch_n,
            where=where,
            include_content=True,
            include_derived=True,
        )
    except Exception as e:
        logger.warning(
            "[CaseSearch] vector_search_v2 失败 collection=%s where=%s: %s",
            collection_id, where, e,
        )
        return []

    cases: list[CaseHit] = []
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        case = _hit_to_case(hit)
        if case is None:
            continue
        if case.cosine_similarity < threshold:
            continue
        cases.append(case)
    cases.sort(key=lambda c: c.cosine_similarity, reverse=True)
    return cases[:top_k]


async def search_cases(
    question: str,
    policy_id: Optional[str],
    *,
    threshold: float,
    top_k: int,
) -> list[CaseHit]:
    """preview 阶段 case 检索主入口（positive / negative 分桶独立召回后合并）。

    对 ``case_polarity`` 做分桶检索：positive 召回 ``top_k``、negative 召回
    ``top_k``，各自先 cosine ≥ ``threshold`` 过滤再按相似度降序截断，最后
    positive 段在前、negative 段在后合并返回（长度 ≤ ``2 * top_k``）。

    meta 解析不出 polarity 列时退化为单次混合检索（取 ``top_k``）。
    任意失败路径都返回 ``[]``，不抛出（preview 自身仍可继续）。
    """

    if top_k <= 0:
        return []
    q = (question or "").strip()
    if not q:
        return []
    collection_id = resolve_case_collection_id(policy_id)
    if not collection_id:
        return []

    q_vec = await _embed_question(q)
    if not q_vec:
        return []

    try:
        from .client import get_default_client
    except Exception as e:  # pragma: no cover
        logger.warning("[CaseSearch] 导入 retrieval client 失败: %s", e)
        return []

    try:
        client = await get_default_client()
    except Exception as e:  # pragma: no cover
        logger.warning("[CaseSearch] 获取 retrieval client 失败: %s", e)
        return []

    polarity_col, tombstone_col = await _resolve_filter_columns(client, collection_id)

    if not polarity_col:
        # 退化路径：拿不到 polarity 列，走单次混合检索。
        result = await _search_one_bucket(
            client, collection_id, q_vec,
            where=None, threshold=threshold, top_k=top_k,
        )
        logger.info(
            "[CaseSearch] collection=%s threshold=%.3f top_k=%d 不分桶命中 %d 条",
            collection_id, threshold, top_k, len(result),
        )
        return result

    buckets: dict[str, list[CaseHit]] = {}
    for polarity in _POLARITIES:
        buckets[polarity] = await _search_one_bucket(
            client, collection_id, q_vec,
            where=_build_where(polarity, polarity_col, tombstone_col),
            threshold=threshold, top_k=top_k,
        )

    result = buckets["positive"] + buckets["negative"]
    logger.info(
        "[CaseSearch] collection=%s threshold=%.3f top_k=%d 命中 positive=%d negative=%d（合计 %d）",
        collection_id, threshold, top_k,
        len(buckets["positive"]), len(buckets["negative"]), len(result),
    )
    return result

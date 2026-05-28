"""preview 阶段 case 库纯向量检索。

供 :mod:`inference.preview` 在生成 preview prompt 之前调用，从 LanceDB v2
集合 ``case_{khCode}`` 拉相关案例经验，按 cosine_similarity 阈值过滤后取 top-k,
注入到 preview user prompt 的【相关案例经验】段。

集合命名约定（与 :mod:`case_refinery` 服务对齐）：

- ``khCode = policyId.split("_")[0]``
- ``collection_id = f"case_{khCode}"``
- inference pipeline 传入的 policy_id 已经被 ``make_index_policy_id`` 拼上
  ``__cs{N}`` 后缀，本模块内部会先剥后缀再 split。

检索协议（详见 :func:`inference.retrieval.client.RetrievalServiceClient.vector_search_v2`）：

- 走 ``POST /v2/collections/{cid}/search``，``query_tokenized=""`` + ``top_m=0``
  纯向量召回；响应 ``hits[*].cosine_similarity`` 直接给出原始 cosine 分。

降级策略（全链路兜底）：

- policy_id 解析不出 kh_code → ``[]``；
- embedding 服务异常 → ``[]``；
- LanceDB 集合不存在（404）→ ``[]``（case_refinery 未上线是预期）；
- 服务不可达 / 5xx → ``[]``；
- 命中 0 条 ≥ 阈值 → ``[]``。

返回 ``[]`` 时 :func:`inference.prompts.select_preview_prompt` 会自动回退到
原 2 套 PREVIEW_* prompt（不渲染【相关案例经验】段），与 ``topC=0`` 体验一致。
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


async def search_cases(
    question: str,
    policy_id: Optional[str],
    *,
    threshold: float,
    top_k: int,
) -> list[CaseHit]:
    """preview 阶段 case 检索主入口。

    返回按 ``cosine_similarity`` 降序的 ``CaseHit`` 列表，长度 ≤ ``top_k``。
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

    # 多召回一些再阈值过滤，避免阈值卡得严时凑不齐 top_k；
    # 服务端最大 top_n 没有硬上限，这里按 3 倍兜底（且至少 +5）。
    fetch_n = max(top_k * 3, top_k + 5)
    try:
        client = await get_default_client()
        hits = await client.vector_search_v2(
            collection_id,
            query_vector=q_vec,
            top_n=fetch_n,
            include_content=True,
            include_derived=True,
        )
    except Exception as e:
        logger.warning(
            "[CaseSearch] vector_search_v2 失败 collection=%s: %s",
            collection_id, e,
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
    result = cases[:top_k]
    if result:
        logger.info(
            "[CaseSearch] collection=%s threshold=%.3f top_k=%d 命中 %d 条（拉取 %d 条）",
            collection_id, threshold, top_k, len(result), len(hits or []),
        )
    else:
        logger.info(
            "[CaseSearch] collection=%s threshold=%.3f 无命中（拉取 %d 条）",
            collection_id, threshold, len(hits or []),
        )
    return result

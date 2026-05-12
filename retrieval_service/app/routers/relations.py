"""关联派生 chunk 反查 / 展开。"""

from __future__ import annotations

import logging
from functools import partial

import anyio
from fastapi import APIRouter, Depends, HTTPException, Query

from .. import store
from ..deps import require_api_key
from ..schema import (
    DependentEntry,
    DependentsResponse,
    ExpandRequest,
    ExpandResponse,
    LookupResponse,
    SearchHit,
)

router = APIRouter(tags=["relations"], dependencies=[Depends(require_api_key)])

logger = logging.getLogger(__name__)


@router.post(
    "/v1/policies/{policy_id}/relations:expand",
    response_model=ExpandResponse,
)
async def expand_relations(policy_id: str, body: ExpandRequest) -> ExpandResponse:
    fn = partial(
        store.expand_relations,
        policy_id,
        body.chunk_id,
        include_content=body.include_content,
    )
    try:
        chunks = await anyio.to_thread.run_sync(fn)
    except Exception as e:
        logger.exception("[relations] expand 失败 policy=%s chunk=%s", policy_id, body.chunk_id)
        raise HTTPException(status_code=500, detail=str(e))
    return ExpandResponse(chunks=chunks)


@router.get(
    "/v1/policies/{policy_id}/relations:lookup",
    response_model=LookupResponse,
)
async def lookup_in_policy(
    policy_id: str,
    target_policy_id: str = Query(..., description="目标外部条款的 policyId"),
    target_clause_id: str | None = Query(default=None, description="可选：限定 clauseId"),
    include_content: bool = Query(default=False),
) -> LookupResponse:
    fn = partial(
        store.lookup_relations,
        policy_id,
        target_policy_id=target_policy_id,
        target_clause_id=target_clause_id,
        include_content=include_content,
    )
    chunks: list[SearchHit] = await anyio.to_thread.run_sync(fn)
    return LookupResponse(chunks=chunks)


@router.get("/v1/relations:lookup-dependents", response_model=DependentsResponse)
async def lookup_dependents(
    target_policy_id: str = Query(..., description="目标外部 policyId"),
    target_clause_id: str | None = Query(default=None),
) -> DependentsResponse:
    """全局反查依赖某 (policy, clause) 的所有源 policy。

    供主项目 cascade 触发器使用：``app.py._cascade_dependent_rebuilds`` 调用本接口
    一次性获得所有需要 force_rebuild 的 source policy。
    """
    fn = partial(store.lookup_dependents, target_policy_id, target_clause_id)
    raw = await anyio.to_thread.run_sync(fn)
    return DependentsResponse(
        dependents=[DependentEntry(source_policy_id=pid, n_hits=n) for pid, n in raw]
    )

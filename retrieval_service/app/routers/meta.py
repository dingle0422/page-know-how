"""表 meta / 列出所有 policy。"""

from __future__ import annotations

import logging

import anyio
from fastapi import APIRouter, Depends, HTTPException

from .. import store
from ..deps import require_api_key
from ..schema import PolicyListItem, PolicyListResponse, TableMeta

router = APIRouter(tags=["meta"], dependencies=[Depends(require_api_key)])

logger = logging.getLogger(__name__)


@router.get("/v1/policies", response_model=PolicyListResponse)
async def list_policies() -> PolicyListResponse:
    raw = await anyio.to_thread.run_sync(store.list_policies)
    return PolicyListResponse(
        policies=[PolicyListItem(policy_id=pid, n_chunks=n, dim=dim) for pid, n, dim in raw]
    )


@router.get("/v1/policies/{policy_id}/meta", response_model=TableMeta | None)
async def get_meta(policy_id: str) -> TableMeta | None:
    data = await anyio.to_thread.run_sync(store.table_meta, policy_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"policy not found: {policy_id}")
    return TableMeta(**data)

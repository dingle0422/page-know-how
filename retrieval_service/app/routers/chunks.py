"""chunks 写入 / 列表 / 删除。"""

from __future__ import annotations

import logging

import anyio
from fastapi import APIRouter, Depends, HTTPException, Query

from .. import store
from ..deps import require_api_key
from ..schema import (
    SearchHit,
    UpsertRequest,
    UpsertResponse,
)

router = APIRouter(
    prefix="/v1/policies/{policy_id}",
    tags=["chunks"],
    dependencies=[Depends(require_api_key)],
)

logger = logging.getLogger(__name__)


@router.post("/chunks:upsert", response_model=UpsertResponse)
async def upsert_chunks(policy_id: str, body: UpsertRequest) -> UpsertResponse:
    try:
        result = await anyio.to_thread.run_sync(
            store.upsert,
            policy_id,
            body.chunks,
            body.mode,
            body.expected_dim,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("[chunks] upsert 失败 policy=%s", policy_id)
        raise HTTPException(status_code=500, detail=f"upsert failed: {e}")
    return UpsertResponse(**result)


@router.get("/chunks", response_model=list[SearchHit])
async def list_chunks(
    policy_id: str,
    where: str | None = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=100000),
    include_content: bool = Query(default=False),
) -> list[SearchHit]:
    try:
        hits = await anyio.to_thread.run_sync(
            store.list_chunks,
            policy_id,
            where,
            limit,
            include_content,
        )
    except Exception as e:
        logger.exception("[chunks] list 失败 policy=%s", policy_id)
        raise HTTPException(status_code=500, detail=str(e))
    return hits


@router.delete("")
async def drop_policy(policy_id: str) -> dict:
    ok = await anyio.to_thread.run_sync(store.drop_table, policy_id)
    return {"ok": bool(ok)}

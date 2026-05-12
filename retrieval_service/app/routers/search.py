"""hybrid 检索路由。

入参由客户端预算好：``query_tokenized``（jieba 分词后空格连接） + ``query_vector``。
服务端只负责调 LanceDB FTS / 向量检索 + 本地 RRF 融合，与主项目 ``inference/retrieval/rrf.py``
完全等价。
"""

from __future__ import annotations

import logging
from functools import partial

import anyio
from fastapi import APIRouter, Depends, HTTPException

from .. import store
from ..config import get_settings
from ..deps import require_api_key
from ..schema import SearchRequest, SearchResponse

router = APIRouter(
    prefix="/v1/policies/{policy_id}",
    tags=["search"],
    dependencies=[Depends(require_api_key)],
)

logger = logging.getLogger(__name__)


@router.post("/search", response_model=SearchResponse)
async def search(policy_id: str, body: SearchRequest) -> SearchResponse:
    settings = get_settings()
    rrf_k = body.rrf_k if body.rrf_k is not None else settings.rrf_k
    fn = partial(
        store.hybrid_search,
        policy_id,
        query_tokenized=body.query_tokenized,
        query_vector=body.query_vector,
        top_n=body.top_n,
        top_m=body.top_m,
        rrf_k=rrf_k,
        where=body.where,
        include_content=body.include_content,
        include_derived=body.include_derived,
    )
    try:
        hits = await anyio.to_thread.run_sync(fn)
    except Exception as e:
        logger.exception("[search] policy=%s 失败", policy_id)
        raise HTTPException(status_code=500, detail=str(e))
    return SearchResponse(hits=hits)

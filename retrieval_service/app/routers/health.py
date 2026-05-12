"""健康检查 + 版本信息。"""

from __future__ import annotations

from fastapi import APIRouter

from .. import __version__

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "version": __version__}

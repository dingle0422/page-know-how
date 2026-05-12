"""HTTP 依赖项：API Key 鉴权。"""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from .config import get_settings


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    settings = get_settings()
    expected = settings.api_key
    if not expected:
        return  # 未配置 API_KEY 时关闭鉴权（仅本地开发）

    candidate = x_api_key or ""
    if not candidate and authorization:
        if authorization.lower().startswith("bearer "):
            candidate = authorization[7:].strip()

    if candidate != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid api key",
        )

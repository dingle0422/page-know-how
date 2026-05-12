"""retrieval_service 主入口。

裸启动：``uvicorn app.main:app --host 0.0.0.0 --port 8088``
容器启动：见 ``Dockerfile``，``docker-compose up`` 即可。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import chunks, health, meta, relations, search


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)
    logger = logging.getLogger("retrieval_service")
    logger.info(
        "retrieval_service starting: store_dir=%s, scalar_index=%s, auth=%s",
        settings.store_dir,
        settings.enable_scalar_index,
        "on" if settings.api_key else "off",
    )
    yield
    logger.info("retrieval_service stopped")


app = FastAPI(
    title="page-know-how retrieval service",
    description="LanceDB 驱动的 BM25 + 向量混合检索微服务（每 policy 一张表）。",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chunks.router)
app.include_router(search.router)
app.include_router(relations.router)
app.include_router(meta.router)

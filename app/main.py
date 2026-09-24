"""FastAPI 应用入口。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.health import router as health_router
from app.api.rag import router as rag_router
from app.config import get_settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """准备本地目录，避免在模块导入阶段连接外部服务。"""

    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    yield


def create_app() -> FastAPI:
    """创建 HTTP 应用。"""

    app = FastAPI(
        title="SRAG",
        version=__version__,
        description="面向课程 Notebook 与 PDF 资料的本地优先 RAG 服务",
        lifespan=lifespan,
    )
    app.include_router(health_router, prefix="/api", tags=["健康检查"])
    app.include_router(rag_router, prefix="/api/rag", tags=["知识检索"])
    return app


app = create_app()
